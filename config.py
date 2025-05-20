import os

# Path configurations
JD_PATH = './dataset/jd'
RESUME_PATH = './dataset/resume'
LORA_PATHS = {
    'meta-llama/Llama-3.2-1B': 'weights/llama3-1B',
    'meta-llama/Llama-3.2-3B': 'weights/llama3-3B',
    'meta-llama/Llama-3.2-8B': 'weights/llama3-8B',
}
DOMAINS = [s.replace('.json', '') for s in os.listdir(JD_PATH)]

# Model configurations
METADATA_MODEL = 'meta-llama/Llama-3.2-1B'
AGENT_MODEL = 'meta-llama/Llama-3.2-1B'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
RAG_K_THRESHOLD = 5
