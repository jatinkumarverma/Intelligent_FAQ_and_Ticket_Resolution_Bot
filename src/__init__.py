"""
Offline Support Bot - Intelligent FAQ & Ticket Resolution System
"""

__version__ = "0.1.0"
__author__ = "Support Bot Team"

from src.config import BASE_DIR, OLLAMA_HOST, OLLAMA_MODEL_NAME
from src.logger import logger

logger.info(f"Support Bot v{__version__} initialized")
