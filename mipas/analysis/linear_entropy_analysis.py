# mipas/analysis/linear_entropy_analysis.py

# ================ Standard library imports ================
import io
import math
from pathlib import Path
from typing import Dict, Tuple, List, Any
import logging

# ================ Third-party library imports  ================
import matplotlib

matplotlib.use(
    "Agg"
)  # Use the 'Agg' backend in headless mode for performance and thread safety.
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import shannon_entropy
from PIL import Image

# ================ Project-specific Imports ================
from mipas.logging_config import setup_worker_logger
from mipas.config.configuration_manager import ConfigurationManager

# ================ Logger Setup ================
logger = logging.getLogger(__name__)


class LSEAConfigurationManager(ConfigurationManager):
    def __init__(self, config: Dict[str, Any], input_file: Path, input_folder: Path, analysis_type: str):
        try:
            # Set output folder explicitly before calling base constructor
            config = config.copy()
            parent_folder = input_folder.parent
            config["output_folder"] = config.get("output_folder") or str(parent_folder / f"{analysis_type}_results_{input_folder.name}")

            super().__init__(config, input_file, input_folder, analysis_type)

            # Get validated config values
            self.sliding_window_entropy_percent = self.cfg["sliding_window_entropy_percent"]
            # Load and convert the image
            image_converter = ImageConverter(input_file)
            self.image = image_converter.convert_to_npy()
            # Use detected bit depth
            self.bit_depth = image_converter.bit_depth
            self.cfg.data["bit_depth"] = self.bit_depth

            # Required for GUI & plot logic
            self.generate_plots = self.cfg["plot"]
            self.plot_figsize = self.cfg["plot_figsize"]
            self.title_size = self.cfg["title_size"]
            self.axis_label_size = self.cfg["axis_label_size"]
            self.tick_label_size = self.cfg["tick_label_size"]
            self.legend_size = self.cfg["legend_size"]
            self.dpi_setting = self.cfg["dpi_setting"]

            logger.info(f"Image converted and bit depth set for {input_file}")
        except Exception as e:
            logger.error(f"Failed to initialize LSEAConfigurationManager: {str(e)}")
            raise


class ImageConverter:
    """
    Converts image files (e.g., .tif, .png) into a numpy array and ensures the image is grayscale.

    Converting to grayscale is critical because Shannon entropy calculations in this context 
    are based on the pixel intensity distribution, which is most meaningful in grayscale images.

    Args:
        image_path (Path): Path to the input image file.

    Returns:
        np.ndarray: Grayscale image data as a numpy array.
        
    Raises:
        FileNotFoundError: If the image file is not found.
        IOError: If there's an issue reading or converting the image file.
    """
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
            if img.mode not in ["L", "I;16"]:
                img = img.convert("L")

            img_array = np.array(img)
            self.bit_depth = img_array.dtype.itemsize * 8
            logger.info(f"[{self.image_path.name}] Converted to numpy array, bit depth = {self.bit_depth}")
            return img_array
        except FileNotFoundError:
            logger.exception(f"[{self.image_path.name}] File not found during image conversion")
            raise
        except IOError:
            logger.exception(f"[{self.image_path.name}] IO error during image conversion")
            raise


class LineEntropyCalculator:
    def __init__(self, img_array: np.ndarray, file_name: str):
        """
        Args:
            img_array (np.ndarray): Numpy array of the image data.
            file_name (str): Name of the file being processed.
        """
        self.img_array = img_array
        self.file_name = file_name

    def calculate_line_entropy(self) -> np.ndarray:
        """
        Calculates line-by-line Shannon entropy for the image.

        Returns:
            np.ndarray: 1D array of entropy values for each line.

        Raises:
            Exception: For errors during entropy calculation.
        """
        try:
            entropy = np.apply_along_axis(shannon_entropy, axis=1, arr=self.img_array)
            logger.info(f"Line entropy calculation completed for file: {self.file_name}")
            return entropy
        except Exception as e:
            logger.error(f"Error calculating line entropy for file {self.file_name}: {str(e)}")
            raise


class slidingWindowEntropyCalculator:
    def __init__(self, img_array: np.ndarray, sliding_window_entropy_percent: float, file_name: str):
        """
        Args:
            img_array (np.ndarray): Numpy array of the image data.
            sliding_window_entropy_percent (float): Percentage size of the sliding window.
            file_name (str): Name of the file being processed.
        """
        self.img_array = img_array
        self.sliding_window_entropy_percent = sliding_window_entropy_percent
        self.file_name = file_name

    def calculate_sliding_window_entropy(self) -> np.ndarray:
        """
        Calculates sliding-window entropy for each row based on a window size percentage.

        Returns:
            np.ndarray: 1D array of sliding window entropy values.

        Raises:
            ValueError: If the sliding window size is invalid.
            Exception: For other errors during entropy calculation.
        """
        try:
            rows = self.img_array.shape[0]
            slice_width = int(rows * self.sliding_window_entropy_percent / 100)
            offset = math.floor(slice_width / 2)
            sliding_window_entropies = np.full((rows,), np.nan)  # Initialize with NaNs

            for start in range(offset, rows - slice_width + offset + 1):
                slice_ = self.img_array[start - offset : start + slice_width - offset, :]
                sliding_window_entropies[start - offset + slice_width // 2] = shannon_entropy(slice_)
            
            logger.info(f"sliding window entropy calculation completed for file: {self.file_name}")
            return sliding_window_entropies
        except ValueError as e:
            logger.error(f"Invalid window size for sliding window entropy in file {self.file_name}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error calculating sliding window entropy for file {self.file_name}: {str(e)}")
            raise


class ReportManager:
    """Superclass for managing report generation."""

    def __init__(self, output_subfolder: Path):
        self.output_subfolder = output_subfolder


class TextReportGenerator(ReportManager):
    """Generates and saves written reports."""

    def generate_report(
        self, line_entropy: np.ndarray, sliding_window_entropy: np.ndarray
    ) -> List[str]:
        """
        Generates the report content based on line and sliding window entropy.

        Args:
            line_entropy (np.ndarray): Line entropy values.
            sliding_window_entropy (np.ndarray): sliding window entropy values.

        Returns:
            List[str]: List of strings representing the report content.

        Raises:
            Exception: If report generation fails.
        """
        try:
            report_content = ["Position\tLine Entropy\tsliding Window Entropy\n"]
            for i, (line_val, sliding_window_val) in enumerate(
                zip(line_entropy, sliding_window_entropy)
            ):
                report_content.append(f"{i}\t{line_val}\t{sliding_window_val}\n")
            return report_content
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

    def save_raw_data(self, filename: Path, line_entropy: np.ndarray, sliding_window_entropy: np.ndarray) -> None:
        """
        Saves the raw entropy data to a CSV file.

        Args:
            filename (Path): Path to save the raw data CSV.
            line_entropy (np.ndarray): Line entropy values.
            sliding_window_entropy (np.ndarray): sliding window entropy values.

        Raises:
            IOError: If there is an issue writing to the file.
        """
        try:
            csv_content = ["Position,Line Entropy,sliding Window Entropy\n"]
            for i, (line_val, sliding_window_val) in enumerate(
                zip(line_entropy, sliding_window_entropy)
            ):
                csv_content.append(f"{i},{line_val},{sliding_window_val}\n")

            with open(filename, "w") as csv_file:
                csv_file.writelines(csv_content)
            logger.info(f"Raw entropy data saved to: {filename}")
        except IOError as e:
            logger.error(f"Error saving raw data to {filename}: {str(e)}")
            raise

class PlotReportGenerator(ReportManager):
    """
    Generates and saves plots/images of results.
    
    Args:
        output_subfolder (Path): Directory to save the plots.
    """

    def plot_entropy_variation(
        self, 
        line_entropy: np.ndarray, 
        sliding_window_entropy: np.ndarray, 
        config: Dict[str, Any],
        output_file_path: Path  # Pass the correct output path
    ) -> None:
        """
        Plots and saves the entropy variation graph.

        Args:
            line_entropy (np.ndarray): Array of line entropy values.
            sliding_window_entropy (np.ndarray): Array of sliding window entropy values.
            config (Dict[str, Any]): Configuration dictionary containing plot settings (e.g., plot size, bit depth, etc.).
            output_file_path (Path): Correct path where the file should be saved.
        
        Raises:
            IOError: If there is an issue saving the plot.
            Exception: For any other unexpected errors.
        """
        try:
            plt.figure(figsize=config["plot_figsize"])
            x_positions = np.arange(len(line_entropy))
            y_max = config["bit_depth"]

            plt.plot(x_positions, line_entropy, label="Line Entropy")
            plt.plot(x_positions, sliding_window_entropy, label="sliding Window Entropy")

            plt.xlabel("Position", fontsize=config["axis_label_size"])
            plt.ylabel("Entropy", fontsize=config["axis_label_size"])
            plt.ylim(0, 1.1 * y_max)
            plt.title(f"Entropy Variation for {config['image_path'].name}", fontsize=config["title_size"])
            plt.xticks(fontsize=config["tick_label_size"])
            plt.yticks(fontsize=config["tick_label_size"])
            plt.legend(fontsize=config["legend_size"])
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format="tif", dpi=config["dpi_setting"])
            plt.close()

            buf.seek(0)
            img = Image.open(buf)
            plot_filename = output_file_path / f"{config['image_path'].stem}_entropy_variation.tif"

            # Save the plot as a TIFF with compression
            with open(plot_filename, 'wb') as plot_file:
                img.save(plot_file, format="TIFF", compression="tiff_lzw")
            
            # Log the correct path for the plot file
            logger.info(f"Plot saved to {plot_filename}")

        except FileNotFoundError as e:
            logger.error(f"File not found during plot generation: {str(e)}")
            raise
        except IOError as e:
            logger.error(f"IO error during plot saving: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error generating plot: {str(e)}")
            raise


class LinearShannonEntropyAnalysisController:
    """
    Calculates Shannon entropy line-by-line for a given image, assuming the image is grayscale.

    Shannon entropy provides a measure of uncertainty or unpredictability within the pixel intensity 
    distribution. The function is applied to each row of the image.

    Args:
        img_array (np.ndarray): Grayscale image data.
        file_name (str): Name of the file being analyzed.

    Returns:
        np.ndarray: Array of entropy values, one for each line of pixels.
    """

    def __init__(self, config: Dict[str, Any], image_path: Path, input_folder: Path, analysis_type: str):
        # Pass analysis_type to BaseConfigurationManager
        try:
            # Update to pass the input_folder and analysis_type to the configuration manager
            self.config_manager = LSEAConfigurationManager(config, image_path, input_folder, analysis_type)
            
        except Exception as e:
            logger.error(f"Error initializing analysis controller for file {image_path.name}: {str(e)}")
            raise


    def run_analysis(self, file_path: Path) -> None:
        """
        Standardized method to run analysis (for compatibility with entry.py).
        """
        self.run_linear_analysis(file_path, self.config_manager.input_folder)


    def run_linear_analysis(self, image_path: Path, input_folder) -> None:
        """
        Runs the linear entropy analysis on the provided image.

        Args:
            image_path (Path): Path to the image file to analyze.

        Raises:
            Exception: For general errors during analysis execution.
        """
        try:
            # Step 1: Prepare the image for analysis
            self._prepare_image(image_path)

            # Step 2: Perform entropy calculations
            line_entropy, sliding_window_entropy = self._perform_entropy_calculations(image_path)

            # Step 3: Save the results (visual and textual reports)
            self._save_results(line_entropy, sliding_window_entropy, image_path)
        except Exception as e:
            logger.error(f"Error running linear analysis for file {image_path.name}: {str(e)}")
            raise

    def _prepare_image(self, image_path: Path) -> None:
        """
        Prepares the image for analysis by loading and converting it.

        Args:
            image_path (Path): Path to the image file.

        Raises:
            ValueError: If the file extension is unsupported.
            Exception: For other errors during image preparation.
        """
        try:
            self.config_manager.filename = image_path.name

            supported_image_extensions = [".png", ".tif", ".jpg", ".jpeg"]
            supported_np_extensions = [".npy"]

            if image_path.suffix.lower() in supported_image_extensions:
                image_converter = ImageConverter(image_path)
                self.config_manager.image = image_converter.convert_to_npy()
                bit_depth = image_converter.bit_depth

            elif image_path.suffix.lower() in supported_np_extensions:
                np_array = np.load(image_path)
                self.config_manager.image = np_array
                bit_depth = np_array.dtype.itemsize * 8

            else:
                raise ValueError(f"Unsupported file extension: {image_path.suffix}. Only images (e.g., JPG, PNG, TIFF) or preprocessed numpy arrays (.npy) are supported.")

            # Store the detected bit depth for downstream use (e.g., plotting)
            self.config_manager.bit_depth = bit_depth
            self.config_manager.cfg.data["bit_depth"] = bit_depth

            logger.info(f"Image prepared for analysis: {image_path}, bit depth: {bit_depth}")

        except ValueError as e:
            logger.error(f"Unsupported file extension: {image_path.suffix} - {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error preparing image {image_path}: {str(e)}")
            raise

    def _perform_entropy_calculations(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates both line entropy and sliding window entropy.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the line entropy and sliding window entropy arrays.

        Raises:
            Exception: For errors during entropy calculations.
        """
        try:
            # Calculate line entropy
            line_calculator = LineEntropyCalculator(self.config_manager.image, image_path.name)
            line_entropy = line_calculator.calculate_line_entropy()

            # Calculate sliding window entropy
            sliding_window_calculator = slidingWindowEntropyCalculator(
                self.config_manager.image, 
                self.config_manager.sliding_window_entropy_percent,
                image_path.name
            )
            sliding_window_entropy = sliding_window_calculator.calculate_sliding_window_entropy()

            logger.info(f"Entropy calculations completed for file: {image_path.name}")
            return line_entropy, sliding_window_entropy

        except Exception as e:
            logger.error(f"Error performing entropy calculations for file {image_path.name}: {str(e)}")
            raise

    def _save_results(self, line_entropy: np.ndarray, sliding_window_entropy: np.ndarray, image_path: Path) -> None:
        """
        Saves the entropy analysis results as visual plots and raw data in CSV format.

        Args:
            line_entropy (np.ndarray): Array of line entropy values.
            sliding_window_entropy (np.ndarray): Array of sliding window entropy values.

        Raises:
            Exception: If there are any errors during saving results.
        """
        try:
            # Get the relative path of the input file within the input folder
            relative_path = image_path.relative_to(self.config_manager.input_folder)

            # Construct the output folder path without duplication
            output_file_path = self.config_manager.output_folder / relative_path.parent
            output_file_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

            # Save visual results
            if self.config_manager.generate_plots:
                visualizer = PlotReportGenerator(self.config_manager.output_folder)
                visualizer.plot_entropy_variation(
                    line_entropy, 
                    sliding_window_entropy, 
                    {
                        "plot_figsize": self.config_manager.plot_figsize,
                        "bit_depth": self.config_manager.bit_depth,
                        "axis_label_size": self.config_manager.axis_label_size,
                        "title_size": self.config_manager.title_size,
                        "tick_label_size": self.config_manager.tick_label_size,
                        "legend_size": self.config_manager.legend_size,
                        "dpi_setting": self.config_manager.dpi_setting,
                        "image_path": Path(self.config_manager.filename)
                    },
                    output_file_path
                )

            # Generate a unique filename for raw entropy data
            raw_data_path = output_file_path / f"{image_path.stem}_raw_entropy_data_linear.csv"

            # Save raw entropy data in CSV format
            reporter = TextReportGenerator(self.config_manager.output_folder)
            reporter.save_raw_data(raw_data_path, line_entropy, sliding_window_entropy)

            logger.info(f"Results saved successfully for file: {self.config_manager.filename}")

        except Exception as e:
            logger.error(f"Error saving results for file {self.config_manager.filename}: {str(e)}")
            raise