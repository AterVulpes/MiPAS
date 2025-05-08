# mipas/logging_config.py

# ============ Standard library imports ============
import logging
import logging.handlers
import multiprocessing
from pathlib import Path

# ============ Shared Queue ============
log_queue = multiprocessing.Queue(-1)

def setup_main_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file_path = log_dir / "mipas_analysis.log"

    # Console and file handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file_path)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

def setup_worker_logger():
    """
    Set up a standalone logger for subprocesses.
    Each worker logs to stdout independently.
    """
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    if root.hasHandlers():
        root.handlers.clear()

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(process)d] %(asctime)s [%(levelname)s] %(message)s')
    console_handler.setFormatter(formatter)

    root.addHandler(console_handler)