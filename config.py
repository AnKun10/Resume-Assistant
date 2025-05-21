import os

# Path configurations
JD_PATH = './dataset/jd'
RESUME_PATH = './dataset/resume'
DOCUMENTS_PATH = './documents.pkl'
TEMP_UPLOAD_DIR = './temp_uploads'
INDEX_PATH = './resume_index'
LORA_PATHS = {
    'meta-llama/Llama-3.2-1B-Instruct': 'weights/llama3-1B',
    'meta-llama/Llama-3.2-3B-Instruct': 'weights/llama3-3B',
    'meta-llama/Llama-3.1-8B-Instruct': 'weights/llama3-8B',
}
DOMAINS = [s.replace('.json', '') for s in os.listdir(JD_PATH)] if os.path.exists(JD_PATH) else []

# Model configurations
METADATA_MODEL = 'meta-llama/Llama-3.2-1B-Instruct'
AGENT_MODEL = 'meta-llama/Llama-3.2-1B-Instruct'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
RAG_K_THRESHOLD = 5
DEFAULT_LLM_MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
AVAILABLE_LLM_MODELS = [
    'meta-llama/Llama-3.1-8B-Instruct', 
    'meta-llama/Llama-3.2-3B-Instruct',
    'meta-llama/Llama-3.2-1B-Instruct'
]

# Memory management
MEMORY_CHUNK_SIZE = 1000  # Number of documents to process at once
MEMORY_LOGGING_INTERVAL = 5  # Log memory usage after processing this many documents

# UI configurations
APP_TITLE = "Resume Assistant"
APP_ICON = "favicon.ico"
WELCOME_MESSAGE = "Hello Bro!"
FAQ_MESSAGE = "Yes sir!"
ABOUT_MESSAGE = "Down hear Bro!"
API_KEY_INFO = "Please add your Huggingface API key to continue. For more information, please refer to this [page](https://huggingface.co/docs/hub/security-tokens)."
API_KEY_ERROR = "The API key is invalid or expired! For more information, please refer to this [page](https://huggingface.co/docs/hub/security-tokens)."
NO_INDEX_WARNING = "No document index available. Please upload a resume or check if the documents.pkl file is available."
