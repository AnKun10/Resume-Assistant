import streamlit as st
import numpy as np
import time
from streamlit_modal import Modal
from langchain_core.messages import AIMessage, HumanMessage
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import HfApi

from config import METADATA_MODEL, EMBEDDING_MODEL, EMBEDDING_DIM
from utils import load_documents_list, extract_doc, initialize_index, set_seed
import chatbot as chatbot_verbosity
from chatbot import ChatBot
from retriever import ResumeRetriever, get_document_nodes
from llama_index.core import Document

set_seed(42)

# Initialize system messages
welcome_message = "Hello Bro!"
faq_message = "Yes sir!"
about_message = "Down hear Bro!"

# Set up the page configuration
st.set_page_config(
    page_title="Resume Assistant",
    page_icon="favicon.ico",
)
st.title("Resume Assistant")

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [AIMessage(content=welcome_message)]

if "embedding_model" not in st.session_state:
    st.session_state["embedding_model"] = EMBEDDING_MODEL
    st.session_state["embedding_dim"] = EMBEDDING_DIM

if "documents" not in st.session_state:
    try:
        document_list = load_documents_list(pkl_path="./documents.pkl")
        st.session_state["documents"] = document_list
        st.session_state["faiss_index"] = initialize_index(
            docs=st.session_state["documents"],
            model_name=st.session_state["embedding_model"],
            model_dim=st.session_state["embedding_dim"]
        )
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        st.session_state["documents"] = []
        st.session_state["faiss_index"] = None

if "rag_pipeline" not in st.session_state and st.session_state.get("faiss_index") is not None:
    st.session_state["rag_pipeline"] = ResumeRetriever(st.session_state["faiss_index"])

if "cur_resume_list" not in st.session_state:
    st.session_state["cur_resume_list"] = []


def upload_file():
    modal = Modal(title="Upload File Error", key="upload_error", max_width=500)
    if st.session_state["uploaded_file"]:
        try:
            tmp_file = f"./{st.session_state['uploaded_file'].name}.pdf"
            with open(tmp_file, "wb") as f:
                f.write(st.session_state["uploaded_file"].getvalue())
        except Exception as e:
            with modal.container():
                st.markdown("Failed to upload your file! Here is the error:")
                st.error(e)
        else:
            try:
                uploaded_doc = extract_doc(tmp_file)
            except Exception as e:
                with modal.container():
                    st.markdown("Failed to extract your file! Here is the error:")
                    st.error(e)
            else:
                with st.toast('Indexing your file, ready in a moment...'):
                    st.session_state["documents"] = [
                        Document(text=uploaded_doc, metadata={"file_name": st.session_state["uploaded_file"].name})]
                    st.session_state["faiss_index"] = initialize_index(
                        docs=st.session_state["documents"],
                        model_name=st.session_state["embedding_model"],
                        model_dim=st.session_state["embedding_dim"],
                    )
                    st.session_state["rag_pipeline"] = ResumeRetriever(st.session_state["faiss_index"])
                    st.success('Successfully index your file!')
    else:
        try:
            documents = load_documents_list(pkl_path="./documents.pkl")
            st.session_state["documents"] = documents
            st.session_state["faiss_index"] = initialize_index(
                docs=st.session_state["documents"],
                model_name=st.session_state["embedding_model"],
                model_dim=st.session_state["embedding_dim"],
            )
            st.session_state["rag_pipeline"] = ResumeRetriever(st.session_state["faiss_index"])
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")


def check_hf_api(api_token: str):
    api = HfApi()
    try:
        user_info = api.whoami(token=api_token)
        return True
    except HfHubHTTPError as e:
        return False


def clear_message():
    st.session_state["cur_resume_list"] = []
    st.session_state["chat_history"] = [AIMessage(content=welcome_message)]


# Sidebar configuration
with st.sidebar:
    st.markdown("# Configuration")

    st.text_input("Huggingface API Key", type="password", key="api_key")
    st.selectbox(
        label="LLM Model",
        options=['meta-llama/Llama-3.1-8B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct',
                 'meta-llama/Llama-3.2-1B-Instruct'],
        placeholder='meta-llama/Llama-3.1-8B-Instruct',
        key="llm_selection",
    )
    st.file_uploader("Upload resumes", type=["pdf"], key="uploaded_file", on_change=upload_file)
    st.button("Clear conversation", on_click=clear_message)

    st.divider()
    st.markdown(faq_message)

    st.divider()
    st.markdown(about_message)

# Message box for user
user_query = st.chat_input("Type your message here...")

# Display messages
for message in st.session_state["chat_history"]:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
    else:
        with st.chat_message("AI"):
            message[0].render(*message[1:])

# API key validation
if not st.session_state["api_key"]:
    st.info(
        "Please add your Huggingface API key to continue. "
        "For more information, please refer to this [page](https://huggingface.co/docs/hub/security-tokens)."
    )
    st.stop()

if not check_hf_api(st.session_state["api_key"]):
    st.error(
        "The API key is invalid or expired! "
        "For more information, please refer to this [page](https://huggingface.co/docs/hub/security-tokens)."
    )
    st.stop()

# Main chat logic
if st.session_state.get("faiss_index") is not None:
    # Initialize retriever & chatbot
    retriever = st.session_state["rag_pipeline"]
    chatbot = ChatBot(st.session_state.get("llm_selection", 'meta-llama/Llama-3.1-8B-Instruct'))

    # Working flow
    if user_query is not None and user_query != "":
        with st.chat_message("Human"):
            st.markdown(user_query)
            st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("AI"):
            start = time.time()
            with st.spinner("Thinking..."):
                document_list = retriever.retrieve_docs(user_query, chatbot)
                query_type = retriever.metadata["query_type"]
                st.session_state["cur_resume_list"] = document_list
                stream_message = chatbot.generate_message_stream(user_query, document_list, [], query_type)
            end = time.time()

            response = st.write_stream(stream_message)

            retriever_message = chatbot_verbosity
            retriever_message.render(document_list, retriever.metadata, float(end - start))

            st.session_state["chat_history"].append(AIMessage(content=response))
            st.session_state["chat_history"].append((retriever_message, document_list, retriever.metadata, end - start))
else:
    st.warning("No document index available. Please upload a resume or check if the documents.pkl file is available.")
