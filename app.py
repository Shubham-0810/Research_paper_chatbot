import streamlit as st
from chains import get_summary_chain, get_qa_chain
from utils import (
    load_and_split_pdf, batch_chunks, create_vector_store_cached, 
    file_hash, clear_cache_for_new_file, create_qa_prompt_cached
)
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
# Ensure async loop exists
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Research Paper Summarizer & Q&A", layout="wide")
st.title("📄 Research Paper Summarizer & Q&A Chatbot")

MAX_REQUESTS_PER_SESSION = 15  
if "requests" not in st.session_state:
    st.session_state.requests = 0

if "show_advanced" not in st.session_state:
    st.session_state.show_advanced = False


def check_limit():
    if st.session_state.requests >= MAX_REQUESTS_PER_SESSION:
        st.warning("You have reached the maximum allowed requests for this session. Please refresh the page.")
        st.stop()
    st.session_state.requests += 1

# ------------------ File Management ------------------
def handle_file_change(new_hash, filename):
    """Handle when a new file is uploaded"""
    # Clear file-specific session state
    keys_to_clear = [
        "current_retriever", "current_llm", "summary_generated"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Clear cached functions for new file
    clear_cache_for_new_file()
    
    # Update session state
    st.session_state.current_file_hash = new_hash
    st.session_state.current_filename = filename
    st.session_state.file_processed = False
    st.session_state.show_advanced = False
    
# ------------------ Main App Logic ------------------
uploaded_file = st.file_uploader("Upload your research paper (PDF)", type=["pdf"], key= 'file_uploader)

if uploaded_file:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error("File too large! Please upload a PDF under 10 MB.")
        st.session_state.file_uploader = None
        st.stop()
    else:
        new_file_hash = file_hash(uploaded_file)
        
        # Check if this is a new file
        is_new_file = (
            "current_file_hash" not in st.session_state or 
            st.session_state.current_file_hash != new_file_hash
        )
        
        if is_new_file:
            handle_file_change(new_file_hash, uploaded_file.name)
            st.success(f"📄 New PDF loaded: {uploaded_file.name}")
        
        # Display current file info
        st.info(f"**Currently loaded:** {st.session_state.current_filename}")
        
        # Process the document (cached by file hash)
        if not st.session_state.get("file_processed", False):
            with st.spinner("🔄 Processing document... This may take a moment for new files."):
                try:
                    # Load and split (cached by hash)
                    split_docs = load_and_split_pdf(new_file_hash, uploaded_file)
                    st.session_state.split_docs = split_docs
                    
                    # Create vector store (cached by hash) 
                    vector_store = create_vector_store_cached(new_file_hash, split_docs)
                    
                    # Create retriever
                    retriever = vector_store.as_retriever(
                        search_type='mmr', 
                        search_kwargs={"k": 4, 'lambda_mult': 0.5}
                    )
                    st.session_state.current_retriever = retriever
                    
                    # Initialize LLM (reuse same instance)
                    if "llm" not in st.session_state:
                        st.session_state.llm = ChatGoogleGenerativeAI(
                            api_key=st.secrets["GOOGLE_API_KEY"],
                            model='gemini-2.5-flash-lite'
                        )
                    
                    st.session_state.file_processed = True
                    st.success(f"✅ Document processed successfully! ({len(split_docs)} chunks created)")
                    
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")
                    st.stop()
    
        # ------------------ Summary Section ------------------
        st.subheader("📜 Generate Summary")
        
        if st.button("Generate Summary", type="primary"):
            check_limit()
            
            with st.spinner("🤖 Generating summary..."):
                try:
                    batched_docs = batch_chunks(st.session_state.split_docs, batch_size=3)
                    
                    # Get or create summary chain
                    if "summary_chain" not in st.session_state:
                        st.session_state.summary_chain = get_summary_chain(st.session_state.llm)
                    
                    summary = st.session_state.summary_chain.run(batched_docs)
                    
                    # Display summary
                    st.markdown("### 📋 Summary")
                    st.markdown(f"**Document:** {st.session_state.current_filename}")
                    st.markdown("---")
                    st.markdown(summary)
                    st.session_state.summary_generated = True
                    
                except Exception as e:
                    st.error(f"❌ Error generating summary: {str(e)}")
    
        # ------------------ Q&A Section ------------------
        st.subheader("💬 Ask Questions")
        
        # Show context info
        st.caption(f"💡 Ask questions about: **{st.session_state.current_filename}**")
        
        # Question input
        user_question = st.text_input(
            "Enter your question:", 
            placeholder="e.g., What is the main contribution of this paper?",
            key="question_input"
        )
        
        ask_button = st.button("🔍 Get Answer", type="secondary")
        
        
        if ask_button and user_question.strip():
            check_limit()
            
            with st.spinner("🔍 Searching for answer..."):
                try:
                    # Get or create QA chain (cached by file hash)
                    qa_prompt = create_qa_prompt_cached(
                        new_file_hash, 
                        st.session_state.current_retriever
                    )
                    qa_chain = qa_prompt | st.session_state.llm
                    
                    # Get answer
                    answer = qa_chain.invoke(user_question)
                    
                    # Display Q&A
                    st.markdown("### 🤔 Question & Answer")
                    st.markdown(f"**Document:** {st.session_state.current_filename}")
                    st.markdown("---")
                    
                    with st.container():
                        st.markdown(f"**❓ Question:** {user_question}")
                        st.markdown(f"**✅ Answer:** {answer.content}")
                    
                except Exception as e:
                    st.error(f"❌ Error getting answer: {str(e)}")
        
        elif ask_button and not user_question.strip():
            st.warning("⚠️ Please enter a question before clicking 'Get Answer'")
    
        # ------------------ Additional Features ------------------
        with st.expander("🔧 Advanced Options", expanded=st.session_state.show_advanced):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Document Stats"):
                    if "split_docs" in st.session_state:
                        docs = st.session_state.split_docs
                        total_chars = sum(len(doc.page_content) for doc in docs)
                        st.write(f"📄 **Chunks:** {len(docs)}")
                        st.write(f"📝 **Total characters:** {total_chars:,}")
                        st.write(f"📈 **Avg chars/chunk:** {total_chars // len(docs) if docs else 0}")
    
                    st.session_state.show_advanced = True
            
            with col2:
                if st.button("🧹 Clear All Data"):
                    # Clear session state
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    # Clear caches
                    clear_cache_for_new_file()
                    st.success("🧹 All data cleared!")
                    st.rerun()
            
            with col3:
                st.write(f"📊 **Requests used:** {st.session_state.requests}/{MAX_REQUESTS_PER_SESSION}")

else:
    # No file uploaded
    st.info("👆 Please upload a PDF research paper to get started!")
    
    if "current_file_hash" in st.session_state:
        for key in ["current_file_hash", "current_filename", "file_processed"]:
            if key in st.session_state:
                del st.session_state[key]
