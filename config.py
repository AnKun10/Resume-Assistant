import os

# Path configurations
JD_PATH = './dataset/jd'
RESUME_PATH = './dataset/resume'
DOMAINS = [s.replace('.json', '') for s in os.listdir(JD_PATH)]

# Model configurations
METADATA_MODEL = 'meta-llama/Llama-3.2-1B'
AGENT_MODEL = 'meta-llama/Llama-3.2-1B'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDING_DIM = 384
RAG_K_THRESHOLD = 5
