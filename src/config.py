import os
from dotenv import load_dotenv

load_dotenv()

# GEMINI API KEY
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemini-2.5-flash"

# Chunking
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 8
CONFIDENCE_THRESHOLD = 0.75

# Paths
DATA_RAW_DIR = "data/raw"
QA_PAIRS_DIR = "data/qa_pairs"
VECTOR_STORE_DIR = "vector_store"
FAISS_INDEX_NAME = "pharma_index"
