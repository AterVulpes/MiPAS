# ================ Standard Library Imports ================
import io
import math
import logging
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np

# ================ Third-party Library Imports ================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================ Project-specific Imports ================
from mipas.logging_config import setup_worker_logger
from mipas.config.configuration_manager import ConfigurationManager

# ================ Logger Setup ================
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------------
# Configuration Manager
# ---------------------------------------------------------------------------------
class CustomAnalysisConfigurationManager(ConfigurationManager):
    def __init__(self, config: Dict[str, Any], input_file: Path, input_folder: Path, analysis_type: str):
        """
        Manages configuration and initial loading for the custom analysis.
        """
        super().__init__(config, input_file, input_folder, analysis_type)

        try:
            # Initialize additional config fields here (example):
            self.parameter_1 = config.get("parameter_1", 10)
            self.parameter_2 = config.get("parameter_2", "default_value")

            # Example: load or preprocess input image or data
            self.image = self.load_image(input_file)

            logger.info(f"CustomAnalysisConfigurationManager initialized for {input_file}")
        except Exception as e:
            logger.error(f"Failed initializing CustomAnalysisConfigurationManager: {str(e)}")
            raise

    def load_image(self, image_path: Path) -> np.ndarray:
        """
        Loads an image or npy file into a numpy array.

        Returns:
            np.ndarray: Loaded image data.
        """
        try:
            if image_path.suffix.lower() == ".npy":
                img = np.load(image_path)
            else:
                raise ValueError(f"Unsupported file type: {image_path.suffix}")
            logger.info(f"Image loaded: {image_path}")
            return img
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            raise

# ---------------------------------------------------------------------------------
# Core Calculator
# ---------------------------------------------------------------------------------
class CustomAnalysisCalculator:
    def __init__(self, img_array: np.ndarray, config_manager: CustomAnalysisConfigurationManager):
        self.img_array = img_array
        self.config_manager = config_manager

    def perform_analysis(self) -> np.ndarray:
        """
        Main computation for the analysis (example operation).
        """
        try:
            result = np.mean(self.img_array, axis=1)  # Example: mean across axis
            logger.info("Custom analysis calculation completed.")
            return result
        except Exception as e:
            logger.error(f"Error during analysis calculation: {str(e)}")
            raise

# ---------------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------------
class CustomReportManager:
    def __init__(self, output_folder: Path):
        self.output_folder = output_folder

    def save_raw_results(self, result_array: np.ndarray, filename: str):
        try:
            output_path = self.output_folder / f"{filename}_results.csv"
            np.savetxt(output_path, result_array, delimiter=",")
            logger.info(f"Saved raw results to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save raw results: {str(e)}")
            raise

    def save_plot(self, result_array: np.ndarray, filename: str):
        try:
            plt.figure(figsize=(10, 6))
            plt.plot(result_array)
            plt.title("Custom Analysis Results")
            plt.xlabel("Position")
            plt.ylabel("Value")
            plt.grid(True)
            output_path = self.output_folder / f"{filename}_plot.png"
            plt.savefig(output_path, dpi=300)
            plt.close()
            logger.info(f"Saved plot to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save plot: {str(e)}")
            raise

# ---------------------------------------------------------------------------------
# Controller Class (the orchestration logic)
# ---------------------------------------------------------------------------------
class CustomAnalysisController:
    def __init__(self, config: Dict[str, Any], input_file: Path, input_folder: Path, analysis_type: str):
        try:
            self.config_manager = CustomAnalysisConfigurationManager(config, input_file, input_folder, analysis_type)
        except Exception as e:
            logger.error(f"Failed initializing CustomAnalysisController: {str(e)}")
            raise

    def run_analysis(self, file_path: Path) -> None:
        """
        Standardized method called by entry.py
        """
        try:
            # Perform analysis
            calculator = CustomAnalysisCalculator(self.config_manager.image, self.config_manager)
            result_array = calculator.perform_analysis()

            # Save outputs
            output_folder = self.config_manager.output_folder
            report_manager = CustomReportManager(output_folder)
            report_manager.save_raw_results(result_array, file_path.stem)
            report_manager.save_plot(result_array, file_path.stem)

            logger.info(f"Custom analysis complete for {file_path.name}")
        except Exception as e:
            logger.error(f"Error running custom analysis for {file_path.name}: {str(e)}")
            raise
