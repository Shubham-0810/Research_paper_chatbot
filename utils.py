from langchain.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os
import tempfile
import asyncio

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
    
google_api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in secrets.toml or environment variables.")

@st.cache_resource
def load_and_split_pdf(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name    
    
    loader = PyMuPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    os.remove(tmp_path)
    return splitter.split_documents(docs)


@st.cache_resource
def create_vector_store_cached(_documents):
    return create_vector_store(_documents)


def create_vector_store(documents):
    if not documents:
        raise ValueError("No documents were passed to create_vector_store.")
    
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model = 'models/gemini-embedding-001',
        google_api_key= google_api_key
    )

    vector_store = None
    batch_size = 20
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i: i+batch_size]

        try:
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)
        except Exception as e:
            print(f"Skipping batch {i//batch_size} due to error: {e}")
    return vector_store


def batch_chunks(chunks, batch_size= 4):
    batched_docs= []
    for i in range(0, len(chunks), batch_size):
        merged_content = " ".join(chunk.page_content for chunk in chunks[i:i+batch_size])
        batched_docs.append(Document(page_content=merged_content))
    return batched_docs


def format_documents(_retrieved_docs):
    context_text = '\n\n'.join(doc.page_content for doc in _retrieved_docs)
    return context_text

@st.cache_resource
def create_qa_prompt(_retriever):
    prompt= PromptTemplate(
        template= """
        You are a helpful assistant.
        Answer ONLY from the provided context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}""",
        input_variables= ['context', 'question']
    )

    parallel_chain = RunnableParallel({
        'context': _retriever | RunnableLambda(format_documents),
        'question': RunnablePassthrough()
    })

    pmt = parallel_chain | prompt
    return pmt


@st.cache_resource
def latex_check_prompt():
    return PromptTemplate(
        input_variables=["text"],
        template="""
        You are an expert in LaTeX and mathematical writing.

        I will give you some text that contains mathematical expressions, possibly in plain parentheses or square brackets.
        Your task:
        - Detect all math expressions (currently inside parentheses like (x+y), or brackets like [ a = b ]).
        - Rewrite **inline math** using `$ ... $` delimiters.
        - Rewrite **standalone/multi-part equations** using display math delimiters: `$$ ... $$`.
        - Ensure all LaTeX compiles correctly in Markdown and that all symbols and special characters display properly.
        - Keep non-math text exactly as it is.
        - Do not remove any math or change its meaning.

        Return only the corrected text, ready for Markdown rendering with LaTeX.

        Here is the text:
        {text}
        """
    )

      
