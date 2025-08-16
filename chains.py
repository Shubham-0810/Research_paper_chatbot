from langchain.chains.summarize import load_summarize_chain
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from utils import create_qa_prompt_cached, latex_check_prompt
import streamlit as st

@st.cache_resource
def get_summary_chain(_llm):
    return load_summarize_chain(
        llm= _llm,
        chain_type= 'map_reduce'
    )

@st.cache_resource
def get_qa_chain(_llm, _retriever):
    parser = StrOutputParser()
    return create_qa_prompt(_retriever) | _llm | parser | RunnableLambda(lambda x: latex_check_prompt().format(text=x)) | _llm | parser

