"""
Configuration and constants for the offline support bot.
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============= Path Configuration =============
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "docs"
SAMPLE_DIR = DATA_DIR / "sample"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"

# Create directories if they don't exist
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

# ============= LLM Configuration =============
# Ollama settings - runs locally, no API key needed
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL", "mistral")  # mistral, llama2, neural-chat, etc.
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# LLM Generation settings
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))  # Lower = more deterministic
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))

# ============= Embeddings Configuration =============
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # Lightweight & free
EMBEDDING_DIMENSION = 384  # Dimension of all-MiniLM-L6-v2
BATCH_EMBEDDING_SIZE = 32  # Process embeddings in batches for efficiency

# ============= Vector Store Configuration =============
CHROMA_DB_PATH = str(KNOWLEDGE_BASE_DIR / "chroma_db")
CHROMA_COLLECTION_NAME = "support_docs"

# ============= Document Processing =============
CHUNK_SIZE = 512  # Tokens per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks
MAX_CHUNKS_PER_DOC = 100

# Supported file types
SUPPORTED_DOC_TYPES = {".txt", ".pdf", ".docx", ".md", ".json"}

# ============= RAG Configuration =============
RAG_K_RETRIEVAL = 5  # Number of documents to retrieve
RAG_SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score
RAG_RELEVANCE_THRESHOLD = 0.7  # For confidence scoring

# ============= Agent Configuration =============
AGENT_TIMEOUT = 60  # Seconds before agent times out
MAX_AGENT_ITERATIONS = 10  # Max tool use iterations per agent

# Classification thresholds
SIMPLE_TICKET_CONFIDENCE_THRESHOLD = 0.8  # If confidence > this, handle directly
COMPLEX_TICKET_ESCALATION_THRESHOLD = 0.5

# ============= Ticket Categorization =============
TICKET_CATEGORIES = {
    "technical": "Technical Support",
    "billing": "Billing & Payments",
    "account": "Account Management",
    "product": "Product Inquiry",
    "bug": "Bug Report",
    "feature": "Feature Request",
    "general": "General",
}

# ============= Logging =============
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = BASE_DIR / "logs" / "support_bot.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ============= Streamlit Configuration =============
STREAMLIT_THEME = "light"
MAX_RECENT_TICKETS_DISPLAY = 10

# ============= Performance Settings =============
CACHE_EMBEDDINGS = True  # Cache generated embeddings
CACHE_RETRIEVAL = True  # Cache retrieval results for identical queries
CACHE_TTL = 3600  # Cache time-to-live in seconds

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

LOGGER = logging.getLogger("support_bot")

print(f"✓ Configuration loaded from {BASE_DIR}")
print(f"  - Ollama: {OLLAMA_HOST}")
print(f"  - Model: {OLLAMA_MODEL_NAME}")
print(f"  - Embeddings: {EMBEDDING_MODEL}")
print(f"  - Vector Store: {CHROMA_DB_PATH}")
