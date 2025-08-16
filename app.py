import streamlit as st
from chains import get_summary_chain, get_qa_chain
from utils import load_and_split_pdf, batch_chunks, create_vector_store_cached
from langchain_google_genai import ChatGoogleGenerativeAI

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Research Paper Summarizer & Q&A", layout="wide")
st.title("📄 Research Paper Summarizer & Q&A Chatbot")

# Step 1: Upload
uploaded_file = st.file_uploader("Upload your research paper (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        # Load & split
        split_docs = load_and_split_pdf(uploaded_file)

        # Create vector store
        vector_store = create_vector_store_cached(split_docs)
        retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={"k": 4, 'lambda_mult': 0.5})

        # LLM
        # llm = ChatOpenAI(
        #     api_key= st.secrets["OPENAI_API_KEY"],
        #     model='openai/gpt-oss-20b:free',
        #     base_url="https://openrouter.ai/api/v1"
        # )

        llm = ChatGoogleGenerativeAI(
            api_key= st.secrets["GOOGLE_API_KEY"],
            model = 'gemini-2.5-flash-lite'
        )

        # Step 2: Summary
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                batched_docs = batch_chunks(split_docs, batch_size=3)
                summary_chain = get_summary_chain(llm)
                summary = summary_chain.run(batched_docs)
                st.subheader("📜 Summary")
                st.write(summary)

        # Step 3: Q&A
        st.subheader("💬 Ask a Question")
        user_q = st.text_input("Type your question here")
        if st.button("Get Answer"):
            with st.spinner("Fetching answer..."):
                qa_chain = get_qa_chain(llm, retriever)
                answer = qa_chain.invoke(user_q)
                st.write(answer)
