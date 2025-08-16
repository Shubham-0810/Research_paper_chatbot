# from langchain.document_loaders import PyMuPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.schema import Document
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import streamlit as st
# import os, hashlib
# import tempfile
# import asyncio

# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())
    
# google_api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
# if not google_api_key:
#     raise ValueError("GOOGLE_API_KEY not found in secrets.toml or environment variables.")

# def file_hash(uploaded_file):
#     return hashlib.sha256(uploaded_file.getbuffer()).hexdigest()

# @st.cache_resource
# def load_and_split_pdf(hash_key, uploaded_file):

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(uploaded_file.getbuffer())
#         tmp_path = tmp_file.name    
    
#     loader = PyMuPDFLoader(tmp_path)
#     docs = loader.load()
#     splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
#     os.remove(tmp_path)
#     return splitter.split_documents(docs)


# @st.cache_resource
# def create_vector_store_cached(hash_key, _documents):
#     return create_vector_store(_documents)


# def create_vector_store(documents):
#     if not documents:
#         raise ValueError("No documents were passed to create_vector_store.")
    
    
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model = 'models/gemini-embedding-001',
#         google_api_key= google_api_key
#     )

#     vector_store = None
#     batch_size = 20
    
#     for i in range(0, len(documents), batch_size):
#         batch = documents[i: i+batch_size]

#         try:
#             if vector_store is None:
#                 vector_store = FAISS.from_documents(batch, embeddings)
#             else:
#                 vector_store.add_documents(batch)
#         except Exception as e:
#             print(f"Skipping batch {i//batch_size} due to error: {e}")
#     return vector_store


# def batch_chunks(chunks, batch_size= 4):
#     batched_docs= []
#     for i in range(0, len(chunks), batch_size):
#         merged_content = " ".join(chunk.page_content for chunk in chunks[i:i+batch_size])
#         batched_docs.append(Document(page_content=merged_content))
#     return batched_docs


# def format_documents(_retrieved_docs):
#     context_text = '\n\n'.join(doc.page_content for doc in _retrieved_docs)
#     return context_text

# @st.cache_resource
# def create_qa_prompt(_retriever):
#     prompt= PromptTemplate(
#         template= """
#         You are a helpful assistant.
#         Answer ONLY from the provided context.
#         If the context is insufficient, just say you don't know.

#         {context}
#         Question: {question}""",
#         input_variables= ['context', 'question']
#     )

#     parallel_chain = RunnableParallel({
#         'context': _retriever | RunnableLambda(format_documents),
#         'question': RunnablePassthrough()
#     })

#     pmt = parallel_chain | prompt
#     return pmt


# @st.cache_resource
# def latex_check_prompt():
#     return PromptTemplate(
#         input_variables=["text"],
#         template="""
#         You are an expert in LaTeX and mathematical writing.

#         I will give you some text that contains mathematical expressions, possibly in plain parentheses or square brackets.
#         Your task:
#         - Detect all math expressions (currently inside parentheses like (x+y), or brackets like [ a = b ]).
#         - Rewrite **inline math** using `$ ... $` delimiters.
#         - Rewrite **standalone/multi-part equations** using display math delimiters: `$$ ... $$`.
#         - Ensure all LaTeX compiles correctly in Markdown and that all symbols and special characters display properly.
#         - Keep non-math text exactly as it is.
#         - Do not remove any math or change its meaning.

#         Return only the corrected text, ready for Markdown rendering with LaTeX.

#         Here is the text:
#         {text}
#         """
#     )




# import streamlit as st
# import tempfile, os, hashlib, asyncio
# from langchain.document_loaders import PyMuPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.schema import Document
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Ensure async loop exists for Google GenAI
# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

# google_api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
# if not google_api_key:
#     raise ValueError("GOOGLE_API_KEY not found in secrets.toml or environment variables.")

# # ------------------ Helper ------------------
# def file_hash(uploaded_file):
#     return hashlib.sha256(uploaded_file.getbuffer()).hexdigest()

# # ------------------ PDF Loading & Splitting ------------------
# @st.cache_resource(show_spinner=False)
# def load_and_split_pdf(_hash_key, uploaded_file) -> list[Document]:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(uploaded_file.getbuffer())
#         tmp_path = tmp_file.name

#     loader = PyMuPDFLoader(tmp_path)
#     docs = loader.load()
#     os.remove(tmp_path)

#     splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
#     return splitter.split_documents(docs)

# # ------------------ Vector Store ------------------
# @st.cache_resource(show_spinner=False)
# def create_vector_store_cached(_hash_key, _documents: list[Document]):
#     embeddings = GoogleGenerativeAIEmbeddings(
#         model='models/gemini-embedding-001',
#         google_api_key=google_api_key
#     )

#     vector_store = None
#     batch_size = 20
#     for i in range(0, len(_documents), batch_size):
#         batch = _documents[i:i+batch_size]
#         try:
#             if vector_store is None:
#                 vector_store = FAISS.from_documents(batch, embeddings)
#             else:
#                 vector_store.add_documents(batch)
#         except Exception as e:
#             print(f"Skipping batch {i//batch_size} due to error: {e}")
#     return vector_store

# # ------------------ Utilities ------------------
# def batch_chunks(chunks, batch_size=4):
#     batched_docs = []
#     for i in range(0, len(chunks), batch_size):
#         merged_content = " ".join(chunk.page_content for chunk in chunks[i:i+batch_size])
#         batched_docs.append(Document(page_content=merged_content))
#     return batched_docs

# def format_documents(retrieved_docs):
#     return "\n\n".join(doc.page_content for doc in retrieved_docs)

# @st.cache_resource(show_spinner=False)
# def create_qa_prompt(_retriever):
#     prompt = PromptTemplate(
#         template="""
#         You are a helpful assistant.
#         Answer ONLY from the provided context.
#         If the context is insufficient, just say you don't know.

#         {context}
#         Question: {question}""",
#         input_variables=['context', 'question']
#     )

#     parallel_chain = RunnableParallel({
#         'context': _retriever | RunnableLambda(format_documents),
#         'question': RunnablePassthrough()
#     })

#     return parallel_chain | prompt

# @st.cache_resource(show_spinner=False)
# def latex_check_prompt():
#     return PromptTemplate(
#         input_variables=["text"],
#         template="""
#         You are an expert in LaTeX and mathematical writing.

#         I will give you some text that contains mathematical expressions, possibly in plain parentheses or square brackets.
#         Your task:
#         - Detect all math expressions (inside parentheses like (x+y) or brackets like [ a = b ]).
#         - Rewrite **inline math** using `$ ... $` delimiters.
#         - Rewrite **standalone/multi-part equations** using display math delimiters: `$$ ... $$`.
#         - Ensure all LaTeX compiles correctly in Markdown and that all symbols and special characters display properly.
#         - Keep non-math text exactly as it is.
#         - Do not remove any math or change its meaning.

#         Return only the corrected text, ready for Markdown rendering with LaTeX.

#         Here is the text:
#         {text}
#         """
#     )





import streamlit as st
import tempfile, os, hashlib, asyncio
from langchain.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ensure async loop exists for Google GenAI
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

google_api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in secrets.toml or environment variables.")

# ------------------ Helper ------------------
def file_hash(uploaded_file):
    """Generate a unique hash for the uploaded file"""
    return hashlib.sha256(uploaded_file.getbuffer()).hexdigest()

# ------------------ PDF Loading & Splitting ------------------
@st.cache_resource(show_spinner=False)
def load_and_split_pdf(_hash_key, uploaded_file) -> list[Document]:
    """
    Load and split PDF - cached by file hash to avoid reprocessing same file
    The _hash_key parameter ensures cache invalidation when file changes
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

    try:
        loader = PyMuPDFLoader(tmp_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = splitter.split_documents(docs)
        
        # Add metadata to track source
        for i, doc in enumerate(split_docs):
            doc.metadata.update({
                'file_hash': _hash_key,
                'chunk_id': i,
                'total_chunks': len(split_docs)
            })
        
        return split_docs
        
    finally:
        # Always clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ------------------ Vector Store ------------------
@st.cache_resource(show_spinner=False)
def create_vector_store_cached(_hash_key, _documents: list[Document]):
    """
    Create vector store - cached by file hash to prevent API exhaustion
    The _hash_key parameter ensures cache invalidation when file changes
    """
    if not _documents:
        raise ValueError("No documents provided to create vector store.")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model='models/gemini-embedding-001',
        google_api_key=google_api_key
    )

    vector_store = None
    batch_size = 20
    successful_batches = 0
    
    for i in range(0, len(_documents), batch_size):
        batch = _documents[i:i+batch_size]
        try:
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)
            successful_batches += 1
        except Exception as e:
            st.warning(f"Skipping batch {i//batch_size + 1} due to error: {e}")
            continue
    
    if vector_store is None:
        raise RuntimeError("Failed to create vector store - all batches failed")
    
    return vector_store

# ------------------ Chain Creation (Cache by hash) ------------------
@st.cache_resource(show_spinner=False)
def create_qa_prompt_cached(_hash_key, _retriever):
    """Create Q&A prompt chain - cached by file hash"""
    prompt = PromptTemplate(
        template="""
        You are a helpful research assistant analyzing academic papers.
        Answer ONLY based on the provided context from the research paper.
        If the context doesn't contain sufficient information to answer the question, say "I don't have enough information in this document to answer that question."
        
        Be specific and cite relevant parts of the text when possible.
        
        Context from the research paper:
        {context}
        
        Question: {question}
        
        Answer:""",
        input_variables=['context', 'question']
    )

    parallel_chain = RunnableParallel({
        'context': _retriever | RunnableLambda(format_documents),
        'question': RunnablePassthrough()
    })

    return parallel_chain | prompt

# ------------------ Utilities (No caching needed) ------------------
def batch_chunks(chunks, batch_size=4):
    """Combine multiple chunks into larger batches for processing"""
    batched_docs = []
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i+batch_size]
        merged_content = " ".join(chunk.page_content for chunk in batch_chunks)
        
        # Preserve metadata from first chunk in batch
        metadata = batch_chunks[0].metadata.copy() if batch_chunks else {}
        metadata.update({
            'batch_start': i,
            'batch_size': len(batch_chunks),
            'is_batched': True
        })
        
        batched_docs.append(Document(
            page_content=merged_content,
            metadata=metadata
        ))
    return batched_docs

def format_documents(retrieved_docs):
    """Format retrieved documents for context"""
    if not retrieved_docs:
        return "No relevant documents found."
    
    formatted_parts = []
    for i, doc in enumerate(retrieved_docs):
        content = doc.page_content.strip()
        if content:
            formatted_parts.append(f"[Document {i+1}]\n{content}")
    
    return "\n\n".join(formatted_parts)

@st.cache_resource(show_spinner=False) 
def latex_check_prompt():
    """Create LaTeX formatting prompt - can be cached as it's static"""
    return PromptTemplate(
        input_variables=["text"],
        template="""
        You are an expert in LaTeX and mathematical writing.

        I will give you some text that contains mathematical expressions, possibly in plain parentheses or square brackets.
        Your task:
        - Detect all math expressions (inside parentheses like (x+y) or brackets like [ a = b ]).
        - Rewrite **inline math** using `$ ... $` delimiters.
        - Rewrite **standalone/multi-part equations** using display math delimiters: `$$ ... $$`.
        - Ensure all LaTeX compiles correctly in Markdown and that all symbols display properly.
        - Keep non-math text exactly as it is.
        - Do not remove any math or change its meaning.

        Return only the corrected text, ready for Markdown rendering with LaTeX.

        Text to process:
        {text}
        """
    )

# ------------------ Cache Management ------------------
def clear_cache_for_new_file():
    """Clear only the cached functions that depend on file content"""
    # Clear specific cached functions
    if hasattr(load_and_split_pdf, 'clear'):
        load_and_split_pdf.clear()
    if hasattr(create_vector_store_cached, 'clear'):
        create_vector_store_cached.clear()
    if hasattr(create_qa_prompt_cached, 'clear'):
        create_qa_prompt_cached.clear()

def validate_documents(documents):
    """Validate that documents are properly formatted"""
    if not documents:
        return False, "No documents provided"
    
    for i, doc in enumerate(documents):
        if not isinstance(doc, Document):
            return False, f"Document {i} is not a Document instance"
        if not doc.page_content.strip():
            return False, f"Document {i} has empty content"
    
    return True, "Documents are valid"
