import streamlit as st
import time
import os
from streamlit_modal import Modal
from langchain_core.messages import AIMessage, HumanMessage
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub import HfApi

from config import (
    EMBEDDING_MODEL, EMBEDDING_DIM, DOCUMENTS_PATH, TEMP_UPLOAD_DIR,
    APP_TITLE, APP_ICON, WELCOME_MESSAGE, FAQ_MESSAGE, ABOUT_MESSAGE,
    API_KEY_INFO, API_KEY_ERROR, NO_INDEX_WARNING, DEFAULT_LLM_MODEL,
    AVAILABLE_LLM_MODELS, INDEX_PATH
)
from utils import load_documents_list, extract_doc, initialize_index, set_seed, load_index
import chatbot as chatbot_verbosity
from chatbot import ChatBot
from retriever import ResumeRetriever
from llama_index.core import Document

set_seed(42)

# Ensure temp upload directory exists
os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)

# Set up the page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
)
st.title(APP_TITLE)

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [AIMessage(content=WELCOME_MESSAGE)]

if "embedding_model" not in st.session_state:
    # Explicitly initialize the embedding model to avoid OpenAI default
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import Settings
    
    embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.embed_model = embed_model
    
    st.session_state["embedding_model"] = EMBEDDING_MODEL
    st.session_state["embedding_dim"] = EMBEDDING_DIM
    

if "documents" not in st.session_state:
    try:
        document_list = load_documents_list(pkl_path=DOCUMENTS_PATH)
        st.session_state["documents"] = document_list
        st.session_state["faiss_index"] = load_index(index_path=INDEX_PATH)
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
            # Create a unique filename in the temp directory
            filename = st.session_state['uploaded_file'].name
            tmp_file = os.path.join(TEMP_UPLOAD_DIR, f"{filename}")
            
            # Save the uploaded file
            with open(tmp_file, "wb") as f:
                f.write(st.session_state["uploaded_file"].getvalue())
        except Exception as e:
            with modal.container():
                st.markdown("Failed to upload your file! Here is the error:")
                st.error(e)
        else:
            try:
                # Extract document text with memory-efficient processing
                uploaded_doc = extract_doc(tmp_file)
                
                # Clean up the temp file after extraction
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
                    
            except Exception as e:
                with modal.container():
                    st.markdown("Failed to extract your file! Here is the error:")
                    st.error(e)
                # Clean up on error too
                if os.path.exists(tmp_file):
                    os.remove(tmp_file)
            else:
                with st.toast('Indexing your file, ready in a moment...'):
                    # Create document with metadata
                    document = Document(
                        text=uploaded_doc, 
                        metadata={"file_name": st.session_state["uploaded_file"].name}
                    )
                    
                    # Clear previous documents to free memory
                    if "documents" in st.session_state:
                        st.session_state["documents"] = None
                    if "faiss_index" in st.session_state:
                        st.session_state["faiss_index"] = None
                    
                    # Force garbage collection to free memory
                    import gc
                    gc.collect()
                    
                    # Create new index with the single document
                    st.session_state["documents"] = [document]
                    st.session_state["faiss_index"] = initialize_index(
                        docs=st.session_state["documents"],
                        model_name=st.session_state["embedding_model"],
                        model_dim=st.session_state["embedding_dim"],
                    )
                    st.session_state["rag_pipeline"] = ResumeRetriever(st.session_state["faiss_index"])
                    st.success('Successfully indexed your file!')
    else:
        try:
            # Load documents from the configured path
            documents = load_documents_list(pkl_path=DOCUMENTS_PATH)
            
            # Clear previous data to free memory
            if "documents" in st.session_state:
                st.session_state["documents"] = None
            if "faiss_index" in st.session_state:
                st.session_state["faiss_index"] = None
                
            # Force garbage collection
            import gc
            gc.collect()
            
            # Set new data
            st.session_state["documents"] = documents
            st.session_state["faiss_index"] = load_index(index_path=INDEX_PATH)
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
    st.session_state["chat_history"] = [AIMessage(content=WELCOME_MESSAGE)]


# Sidebar configuration
with st.sidebar:
    st.markdown("# Configuration")

    st.text_input("Huggingface API Key", type="password", key="api_key")
    st.selectbox(
        label="LLM Model",
        options=AVAILABLE_LLM_MODELS,
        placeholder=DEFAULT_LLM_MODEL,
        key="llm_selection",
    )
    st.checkbox("Fine-tune Version", key="finetune")
    st.file_uploader("Upload resumes", type=["pdf"], key="uploaded_file", on_change=upload_file)
    st.button("Clear conversation", on_click=clear_message)

    st.divider()
    st.markdown(FAQ_MESSAGE)

    st.divider()
    st.markdown(ABOUT_MESSAGE)

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
    st.info(API_KEY_INFO)
    st.stop()

if not check_hf_api(st.session_state["api_key"]):
    st.error(API_KEY_ERROR)
    st.stop()

# Main chat logic
if st.session_state["faiss_index"]:
    # Initialize retriever & chatbot
    retriever = st.session_state["rag_pipeline"]
    chatbot = ChatBot(
        path=st.session_state["llm_selection"],
        fine_tune=st.session_state["finetune"]
    )

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
    st.warning(NO_INDEX_WARNING)
