# import streamlit as st
# from chains import get_summary_chain, get_qa_chain
# from utils import load_and_split_pdf, batch_chunks, create_vector_store_cached, file_hash
# from langchain_google_genai import ChatGoogleGenerativeAI
# import asyncio

# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())
    
# # ---------- Streamlit UI ----------
# st.set_page_config(page_title="Research Paper Summarizer & Q&A", layout="wide")
# st.title("📄 Research Paper Summarizer & Q&A Chatbot")

# MAX_REQUESTS_PER_SESSION = 5

# if "requests" not in st.session_state:
#     st.session_state.requests = 0

# def check_limit():
#     if st.session_state.requests >= MAX_REQUESTS_PER_SESSION:
#         st.warning("You have reached the maximum allowed requests for this session.")
#         st.stop()
#     st.session_state.requests += 1


# # Step 1: Upload
# uploaded_file = st.file_uploader("Upload your research paper (PDF)", type=["pdf"])

# if uploaded_file:
#     hash_key = file_hash(uploaded_file)
    
#     with st.spinner("Processing document..."):
        
#         # Load & split
#         split_docs = load_and_split_pdf(hash_key, uploaded_file)

#         # Create vector store
#         vector_store = create_vector_store_cached(hash_key,split_docs)
#         retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={"k": 4, 'lambda_mult': 0.5})

#         # LLM
#         # llm = ChatOpenAI(
#         #     api_key= st.secrets["OPENAI_API_KEY"],
#         #     model='openai/gpt-oss-20b:free',
#         #     base_url="https://openrouter.ai/api/v1"
#         # )

#         llm = ChatGoogleGenerativeAI(
#             api_key= st.secrets["GOOGLE_API_KEY"],
#             model = 'gemini-2.5-flash-lite'
#         )

#         # Step 2: Summary
#         if st.button("Generate Summary"):
#             with st.spinner("Summarizing..."):
#                 batched_docs = batch_chunks(split_docs, batch_size=3)
#                 summary_chain = get_summary_chain(llm)
#                 summary = summary_chain.run(batched_docs)
#                 st.subheader("📜 Summary")
#                 st.write(summary)

#         # Step 3: Q&A
#         st.subheader("💬 Ask a Question")
#         user_q = st.text_input("Type your question here")
#         if st.button("Get Answer"):
#             with st.spinner("Fetching answer..."):
#                 qa_chain = get_qa_chain(llm, retriever)
#                 answer = qa_chain.invoke(user_q)
#                 st.write(answer)


import streamlit as st
from chains import get_summary_chain, get_qa_chain
from utils import load_and_split_pdf, batch_chunks, create_vector_store_cached, file_hash
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio

# Ensure async loop exists
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Research Paper Summarizer & Q&A", layout="wide")
st.title("📄 Research Paper Summarizer & Q&A Chatbot")

MAX_REQUESTS_PER_SESSION = 5
if "requests" not in st.session_state:
    st.session_state.requests = 0

def check_limit():
    if st.session_state.requests >= MAX_REQUESTS_PER_SESSION:
        st.warning("You have reached the maximum allowed requests for this session.")
        st.stop()
    st.session_state.requests += 1

# ------------------ File Upload ------------------
uploaded_file = st.file_uploader("Upload your research paper (PDF)", type=["pdf"])
if uploaded_file:
    hash_key = file_hash(uploaded_file)

    with st.spinner("Processing document..."):
        # Load & split
        split_docs = load_and_split_pdf(hash_key, uploaded_file)

        # Create vector store
        vector_store = create_vector_store_cached(hash_key, split_docs)
        retriever = vector_store.as_retriever(search_type='mmr', search_kwargs={"k": 4, 'lambda_mult': 0.5})

        # LLM
        llm = ChatGoogleGenerativeAI(
            api_key=st.secrets["GOOGLE_API_KEY"],
            model='gemini-2.5-flash-lite'
        )

        # ------------------ Summary ------------------
        if st.button("Generate Summary"):
            check_limit()
            with st.spinner("Summarizing..."):
                batched_docs = batch_chunks(split_docs, batch_size=3)
                summary_chain = get_summary_chain(llm)
                summary = summary_chain.run(batched_docs)
                st.subheader("📜 Summary")
                st.write(summary)

        # ------------------ Q&A ------------------
        st.subheader("💬 Ask a Question")
        user_q = st.text_input("Type your question here")
        if st.button("Get Answer"):
            check_limit()
            with st.spinner("Fetching answer..."):
                qa_chain = get_qa_chain(llm, retriever)
                answer = qa_chain.invoke(user_q)
                st.write(answer)
