# mipas/analysis/fd_box_analysis.py

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from mipas.config.configuration_manager import ConfigurationManager

logger = logging.getLogger(__name__)

class ImageConverter:
    """Converts image files (e.g., .tif, .png) to .npy format and ensures they are grayscale."""

    def __init__(self, image_path: Path):
        """
        Args:
            image_path (Path): Path to the image file.
        """
        self.image_path = image_path
        self.bit_depth = None

    def convert_to_npy(self) -> np.ndarray:
        """
        Converts the image to a numpy array and ensures it's grayscale.
        Determines its bit depth based on the image array's data type.

        Returns:
            np.ndarray: The image data in numpy array format.

        Raises:
            FileNotFoundError: If the image file is not found.
            IOError: If there is an issue reading the image file.
        """
        try:
            img = Image.open(self.image_path)
            # Ensure grayscale conversion for SEM images, as they are inherently grayscale.
            if img.mode not in ["L", "I;16"]:
                img = img.convert("L")  # Convert to grayscale to prevent color skew in entropy

            img_array = np.array(img)
            self.bit_depth = img_array.dtype.itemsize * 8  # Determine bit depth based on dtype
            logger.info(f"Image converted to numpy array with bit depth: {self.bit_depth} for {self.image_path}")
            return img_array
        except FileNotFoundError as e:
            logger.error(f"File not found: {self.image_path} - {str(e)}")
            raise
        except IOError as e:
            logger.error(f"IO error while reading image {self.image_path}: {str(e)}")
            raise

class BoxFDConfigurationManager(ConfigurationManager):
    def __init__(self, config: Dict[str, Any], input_file: Path, input_folder: Path, analysis_type: str):
        super().__init__(config, input_file, input_folder, analysis_type)

        # Custom settings from schema-backed config
        self.box_sizes = self.cfg["box_sizes"]
        self.binarize = self.cfg["binarize"]
        self.threshold = self.cfg["threshold"]

        # Load and process image
        image_converter = ImageConverter(input_file)
        self.image = image_converter.convert_to_npy()
        self.bit_depth = image_converter.bit_depth
        self.cfg.data["bit_depth"] = self.bit_depth  # for plotting range use

        logger.info(f"Loaded image: {input_file.name}, bit depth: {self.bit_depth}")


class BoxCountingFractalCalculator:
    def __init__(self, image: np.ndarray, config: BoxFDConfigurationManager):
        self.image = image
        self.config = config

    def calculate_fractal_dimension(self) -> Tuple[float, list, list]:
        binary_image = self.image > self.config.threshold if self.config.binarize else self.image
        box_sizes = self.config.box_sizes
        counts = []

        for size in box_sizes:
            S = (binary_image.shape[0] // size) * size
            cropped = binary_image[:S, :S]
            blocks = cropped.reshape(S // size, size, -1, size).max(axis=(1, 3))
            count = np.sum(blocks)
            counts.append(count if count > 0 else 1)

        logs = [(np.log(1.0 / s), np.log(c)) for s, c in zip(box_sizes, counts)]
        log_r, log_N = zip(*logs)
        coeffs = np.polyfit(log_r, log_N, 1)
        return coeffs[0], log_r, log_N


class BoxCountingFDAnalysisController:
    def __init__(self, config: Dict[str, Any], image_path: Path, input_folder: Path, analysis_type: str):
        self.config_manager = BoxFDConfigurationManager(config, image_path, input_folder, analysis_type)

    def run_analysis(self, file_path: Path) -> None:
        self.run_box_fd_analysis(file_path)

    def run_box_fd_analysis(self, image_path: Path) -> None:
        try:
            fd_calc = BoxCountingFractalCalculator(self.config_manager.image, self.config_manager)
            fd, log_r, log_N = fd_calc.calculate_fractal_dimension()
            self._save_results(image_path, fd, log_r, log_N)
        except Exception as e:
            logger.error(f"Error in box FD analysis for {image_path.name}: {str(e)}")
            raise

    def _save_results(self, image_path: Path, fd_value: float, log_r, log_N) -> None:
        output_dir = self.config_manager.output_folder / image_path.parent.name
        output_dir.mkdir(parents=True, exist_ok=True)

        summary_path = output_dir / "fd_box_summary.txt"
        with open(summary_path, "a", encoding="utf-8") as f:
            rel_path = image_path.relative_to(self.config_manager.input_folder)
            f.write(f"{rel_path.as_posix()}\t{fd_value:.4f}\n")

        plt.figure(figsize=self.config_manager.cfg["plot_figsize"])
        plt.plot(log_r, log_N, 'o-', label=f"FD: {fd_value:.4f}")
        plt.xlabel("log(1/box size)", fontsize=self.config_manager.cfg["axis_label_size"])
        plt.ylabel("log(box count)", fontsize=self.config_manager.cfg["axis_label_size"])
        plt.title(image_path.name, fontsize=self.config_manager.cfg["title_size"])
        plt.xticks(fontsize=self.config_manager.cfg["tick_label_size"])
        plt.yticks(fontsize=self.config_manager.cfg["tick_label_size"])
        plt.legend(fontsize=self.config_manager.cfg["legend_size"])
        plt.tight_layout()

        plot_path = output_dir / f"{image_path.stem}_fd_box_plot.tif"
        plt.savefig(plot_path, dpi=self.config_manager.cfg["dpi_setting"], format="tiff")
        plt.close()

        logger.info(f"FD plot saved to {plot_path}")
