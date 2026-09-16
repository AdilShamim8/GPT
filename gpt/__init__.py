"""
nano-gpt-prod: A production-grade, extensible GPT language model framework built from scratch in PyTorch.
"""

__version__ = "1.0.0"
__author__ = "Adil Shamim"
__license__ = "MIT"

import logging

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger("gpt")
