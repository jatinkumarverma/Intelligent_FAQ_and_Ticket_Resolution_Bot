"""
Logger setup for the support bot.
"""

import logging
import logging.handlers
from pathlib import Path
from src.config import LOG_LEVEL, LOG_FILE

# Create logs directory
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Configure logger
logger = logging.getLogger("support_bot")
logger.setLevel(getattr(logging, LOG_LEVEL))

# File handler
file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE,
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(message)s")
)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)
