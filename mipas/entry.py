# mipas/entry.py

# ============ Standard library imports ============
import os
import json
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Dict, Tuple, Any

# ============ Project-specific imports ============
from mipas.logging_config import setup_worker_logger
logger = logging.getLogger(__name__)

# Import Controllers (each should handle config validation internally)
from mipas.analysis.segmented_entropy_analysis import SegmentedShannonEntropyAnalysisController as SSEAC
from mipas.analysis.linear_entropy_analysis import LinearShannonEntropyAnalysisController as LSEAC
from mipas.analysis.fd_box_analysis import BoxCountingFDAnalysisController as BCFDAC
from mipas.analysis.fd_wtmm_analysis import WTMMFDAnalysisController as WFDAC

# Analysis type to Controller Class mapping
analysis_mapping = {
    "segmented": SSEAC,
    "linear": LSEAC,
    "fd-box": BCFDAC,
    "fd-wtmm": WFDAC,
}

def get_supported_files(input_folder: Path, supported_extensions: list) -> list:
    return [file for file in input_folder.rglob("*") if file.suffix.lower() in supported_extensions]

def process_file(file_path: Path, config: Dict[str, Any], analysis_class: Any, input_folder: Path, analysis_type: str, max_retries: int = 1, current_file: int = 1, total_files: int = 1) -> None:
    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Processing {file_path} ({current_file}/{total_files}), attempt {retries + 1}")
            analysis = analysis_class(config, file_path, input_folder, analysis_type)
            analysis.run_analysis(file_path)
            logger.info(f"Successfully processed {file_path} ({current_file}/{total_files})")
            break
        except Exception:
            retries += 1
            logger.exception(f"Error processing {file_path} ({current_file}/{total_files}), attempt {retries}")
            if retries == max_retries:
                logger.error(f"Failed to process {file_path} after {max_retries} attempts. Skipping.")

def process_file_worker(args: Tuple[Path, Dict[str, Any], str, Path, int, int, int]) -> None:
    file_path, config, analysis_type, input_folder, max_retries, current_file, total_files = args
    analysis_class = analysis_mapping[analysis_type]
    process_file(file_path, config, analysis_class, input_folder, analysis_type, max_retries, current_file, total_files)
    print(f"Worker PID {os.getpid()} completed.")

def run_analysis(config: dict, analysis_type: str = "linear", num_workers: int = -1):
    input_folder = Path(config.get("input_folder"))
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")

    supported_extensions = [".png", ".tif", ".jpg", ".jpeg", ".npy"]
    files_to_analyze = get_supported_files(input_folder, supported_extensions)

    if not files_to_analyze:
        raise ValueError("No supported files found.")

    total_files = len(files_to_analyze)
    if num_workers == -1:
        num_workers = cpu_count()

    max_retries = config.get("max_retries", 3)
    analysis_class = analysis_mapping.get(analysis_type)
    if not analysis_class:
        raise ValueError(f"Unknown analysis type: {analysis_type}")

    def clean_config(config: dict) -> dict:
        """Ensure config contains only basic types for pickling."""
        return json.loads(json.dumps(config))  # strips anything unpicklable

    task_args = [
        (file, clean_config(config), analysis_type, input_folder, max_retries, i + 1, total_files)
        for i, file in enumerate(files_to_analyze)
    ]

    pool = Pool(processes=num_workers, initializer=setup_worker_logger)
    try:
        pool.map(process_file_worker, task_args)
        pool.close()
        pool.join()
        print(f"All workers completed.")
    except Exception as e:
        logger.exception(f"Error during multiprocessing execution: {e}")
        pool.terminate()
        raise
