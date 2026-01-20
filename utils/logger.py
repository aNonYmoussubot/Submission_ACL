# utils/logger.py
import logging
import sys

def setup_logger(name="TrustTable"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger