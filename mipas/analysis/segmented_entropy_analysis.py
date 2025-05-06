# mipas/analysis/segmented_entropy_analysis.py

# ================ Standard library imports ================
from pathlib import Path
from abc import ABC
from typing import List, Tuple, Dict, Any
import logging

# ================ Third-party library imports ================
import matplotlib

matplotlib.use(
    "Agg"
)  # Use the 'Agg' backend in headless mode for performance and thread safety.
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from skimage.measure import shannon_entropy
from PIL import Image

# ================ Project-specific Imports ================
from mipas.logging_config import log_queue
from mipas.config.configuration_manager import ConfigurationManager

# Set up logging
logger = logging.getLogger(__name__)

class SSEAConfigurationManager(ConfigurationManager):
    """
    Manages the configuration for segmented Shannon entropy analysis, including validation and segmentation-specific settings.

    Args:
        config (Dict[str, Any]): User-provided configuration dictionary.
        input_file (Path): Path to the input image.
        input_folder (Path): Directory containing the input file.
        analysis_type (str): Specifies the type of analysis ('Segmented').
    """

    def __init__(self, config: Dict[str, Any], input_file: Path, input_folder: Path, analysis_type: str):
        try:
            config = config.copy()
            parent_folder = input_folder.parent
            config["output_folder"] = config.get("output_folder") or str(parent_folder / f"{analysis_type}_results_{input_folder.name}")

            super().__init__(config, input_file, input_folder, analysis_type)

            # Pull validated values from schema
            self.segmentation_number = self.cfg["segmentation_number"]
            self.segmentation_method = self.cfg["segmentation_method"]
            self.segmentation_cbar_text_size = self.cfg["segmentation_cbar_text_size"]
            self.segmentation_vmin = self.cfg["segmentation_vmin"]
            self.generate_plots = self.cfg["plot"]  # ✅ Fix: ensure this is always available

            # Load and convert image
            image_converter = ImageConverter(input_file)
            self.image = image_converter.convert_to_npy()
            self.bit_depth = image_converter.bit_depth
            self.segmentation_vmax = self.cfg.get("segmentation_vmax", self.bit_depth)

            logger.info(f"Image loaded for SSEA: {input_file.name} (bit depth: {self.bit_depth})")

        except Exception as e:
            logger.error(f"Failed to initialize SSEAConfigurationManager: {e}")
            raise
        
class ReportManager:
    """Base class for managing different types of report generation.
    
    This class provides common functionality for report managers, 
    such as ensuring output directories exist.
    """
    def __init__(self, output_subfolder: Path):
        self.output_subfolder = output_subfolder

    def ensure_output_folder(self):
        """Ensures that the output path exists."""
        self.output_subfolder.mkdir(parents=True, exist_ok=True)


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


class SegmentCalculator(ABC):
    """
    Abstract base class for segmenting images into smaller regions for entropy analysis.
    Different segmentation strategies are implemented in subclasses.

    Segmentation techniques:
    - `CropSegmentCalculator`: Drops incomplete segments that don’t fit within image bounds.
    - `MergeSegmentCalculator`: Merges remainder pixels with the second-to-last row/column.
    - `AdaptiveSegmentCalculator`: Creates smaller segments to fit within the image boundaries.

    Args:
        segment_size (int): The size of the square segment (number of pixels along one side).
        h (int): Image height in pixels.
        w (int): Image width in pixels.
    """
    def calculate_segment_grid(self, segments: int, h: int, w: int) -> List[Dict[str, Any]]:
        num_rows = int(np.sqrt(segments))
        num_cols = (segments + num_rows - 1) // num_rows
        
        segment_height = h // num_rows
        segment_width = w // num_cols
        
        grid = []
        for row in range(num_rows):
            for col in range(num_cols):
                start_i, start_j = row * segment_height, col * segment_width
                end_i, end_j = min((row + 1) * segment_height, h), min((col + 1) * segment_width, w)
                grid.append({'start_i': start_i, 'start_j': start_j, 'end_i': end_i, 'end_j': end_j})
        return grid
    
    def extract_segment(
        self, image: np.ndarray, start_i: int, start_j: int, end_i: int, end_j: int
    ) -> np.ndarray:
        """Extracts a segment of the image based on the start and end points."""
        return image[start_i:end_i, start_j:end_j]

    def calculate_segment_dimensions(
        self, segments: int, h: int, w: int
    ) -> Tuple[int, int]:
        """Calculates the dimensions of each segment based on the number of segments."""
        min_dimension = min(h, w)
        segment_size = min_dimension // segments
        return segment_size, segment_size


class CropSegmentCalculator(SegmentCalculator):
    def calculate_segment_grid(self, segment_size: int, h: int, w: int, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Calculates a grid of square segments where the side length of each segment is the segmentation number.
        This version drops any incomplete segments that do not fully fit within the image bounds.

        Args:
            segment_size: The side length of each square segment (equal to the segmentation number).
            h: The height of the image.
            w: The width of the image.
            image: The image data as a NumPy array.

        Returns:
            A list of dictionaries, where each dictionary contains the pixel data and grid addressing information.
        """
        grid = []

        # Number of complete segments along height and width
        num_rows = h // segment_size
        num_cols = w // segment_size

        # Loop through all rows and columns, excluding segments that do not fit fully in the image
        for row in range(num_rows):
            for col in range(num_cols):
                # Calculate the start and end indices for each square segment
                start_i = row * segment_size
                start_j = col * segment_size
                end_i = start_i + segment_size
                end_j = start_j + segment_size

                # Ensure the segment fits within the image boundaries
                if end_i <= h and end_j <= w:
                    # Extract the current segment from the image
                    current_segment = self.extract_segment(image, start_i, start_j, end_i, end_j)

                    # Append the segment's information and pixel data
                    grid.append({
                        'start_i': start_i,
                        'start_j': start_j,
                        'end_i': end_i,
                        'end_j': end_j,
                        'data': current_segment
                    })
                else:
                    # Log or ignore the dropped segment
                    logger.debug(f"Dropped segment at row {row}, col {col} due to size limitations.")

        return grid


class MergeSegmentCalculator(SegmentCalculator):
    def calculate_segment_grid(self, segment_size: int, h: int, w: int, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Calculates a grid of square segments where the side length of each segment is the segmentation number.
        Handles remainder pixels by merging them with the second-to-last row/column, so the last segment is larger.

        Args:
            segment_size: The side length of each square segment (equal to the segmentation number).
            h: The height of the image.
            w: The width of the image.
            image: The image data as a NumPy array.

        Returns:
            A list of dictionaries, where each dictionary contains the pixel data and grid addressing information.
        """
        grid = []

        # Number of complete segments along height and width
        num_rows = h // segment_size
        num_cols = w // segment_size

        # Remainder pixels for the last row and column
        remainder_height = h % segment_size
        remainder_width = w % segment_size

        # Adjust the number of rows and columns if there's a remainder (we'll merge it with the previous row/col)
        if remainder_height > 0:
            num_rows -= 1  # Merge the remainder with the second-to-last row
        if remainder_width > 0:
            num_cols -= 1  # Merge the remainder with the second-to-last column

        # Loop through all rows and columns except the last (which will handle the remainder merging)
        for row in range(num_rows + 1):  # Includes the last row
            for col in range(num_cols + 1):  # Includes the last column
                # Calculate the start and end indices for each square segment
                start_i = row * segment_size
                start_j = col * segment_size

                # Determine the end indices, adjusting for the remainder pixels in the second-to-last row/column
                if row == num_rows:  # Second-to-last row, merge remainder height
                    end_i = start_i + segment_size + remainder_height  # Merge remainder height
                else:
                    end_i = start_i + segment_size

                if col == num_cols:  # Second-to-last column, merge remainder width
                    end_j = start_j + segment_size + remainder_width  # Merge remainder width
                else:
                    end_j = start_j + segment_size

                # Extract the current segment from the image
                current_segment = self.extract_segment(image, start_i, start_j, end_i, end_j)

                # Append the segment's information and pixel data
                grid.append({
                    'start_i': start_i,
                    'start_j': start_j,
                    'end_i': end_i,
                    'end_j': end_j,
                    'data': current_segment
                })

        return grid


class AdaptiveSegmentCalculator(SegmentCalculator):
    def calculate_segment_grid(self, segment_size: int, h: int, w: int, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Segments the entire image, ensuring that no parts of the image are left uncovered.
        If a segment would extend beyond the image, a smaller segment is created to fit within the boundaries.

        Args:
            segment_size: The side length of each square segment.
            h: The height of the image.
            w: The width of the image.
            image: The image data as a NumPy array.

        Returns:
            A list of dictionaries, where each dictionary contains the pixel data and grid addressing information.
        """
        grid = []

        # Number of full segments along height and width
        num_rows = h // segment_size
        num_cols = w // segment_size

        # Remainder pixels for the last row and column
        remainder_height = h % segment_size
        remainder_width = w % segment_size

        # Loop through rows and columns to create full segments
        for row in range(num_rows):
            for col in range(num_cols):
                # Calculate the start and end indices for each full square segment
                start_i = row * segment_size
                start_j = col * segment_size
                end_i = start_i + segment_size
                end_j = start_j + segment_size

                # Extract the full-sized segment
                current_segment = self.extract_segment(image, start_i, start_j, end_i, end_j)

                # Append the segment's information and pixel data
                grid.append({
                    'start_i': start_i,
                    'start_j': start_j,
                    'end_i': end_i,
                    'end_j': end_j,
                    'data': current_segment
                })

        # Handle any remaining rows (partial row at the bottom)
        if remainder_height > 0:
            for col in range(num_cols):
                start_i = num_rows * segment_size
                start_j = col * segment_size
                end_i = h  # Use the image height for the last segment
                end_j = start_j + segment_size

                # Extract and append the smaller row segment
                current_segment = self.extract_segment(image, start_i, start_j, end_i, end_j)
                grid.append({
                    'start_i': start_i,
                    'start_j': start_j,
                    'end_i': end_i,
                    'end_j': end_j,
                    'data': current_segment
                })

        # Handle any remaining columns (partial column on the right)
        if remainder_width > 0:
            for row in range(num_rows):
                start_i = row * segment_size
                start_j = num_cols * segment_size
                end_i = start_i + segment_size
                end_j = w  # Use the image width for the last segment

                # Extract and append the smaller column segment
                current_segment = self.extract_segment(image, start_i, start_j, end_i, end_j)
                grid.append({
                    'start_i': start_i,
                    'start_j': start_j,
                    'end_i': end_i,
                    'end_j': end_j,
                    'data': current_segment
                })

        # Handle the bottom-right corner (partial segment if both height and width have remainders)
        if remainder_height > 0 and remainder_width > 0:
            start_i = num_rows * segment_size
            start_j = num_cols * segment_size
            end_i = h  # Use the image height
            end_j = w  # Use the image width

            # Extract and append the smaller bottom-right corner segment
            current_segment = self.extract_segment(image, start_i, start_j, end_i, end_j)
            grid.append({
                'start_i': start_i,
                'start_j': start_j,
                'end_i': end_i,
                'end_j': end_j,
                'data': current_segment
            })

        return grid


class EntropyAnalyzer:
    """
    Analyzes image segments and calculates Shannon entropy for each segment.

    Args:
        segment_calculator (SegmentCalculator): The segmentation method to use.

    Methods:
        analyze: Calculates Shannon entropy for a given image segment.
        calculate_segment_entropies: Returns a list of entropy values for all segments in the image.
    """


    def __init__(self, segment_calculator: SegmentCalculator):
        self.segment_calculator: SegmentCalculator = segment_calculator

    @staticmethod
    def analyze(image_segment: np.ndarray) -> float:
        return shannon_entropy(image_segment)
        
    def calculate_segment_entropies(
        self, image: np.ndarray, segments: int, h: int, w: int
    ) -> List[Dict[str, Any]]:
        entropies: List[Dict[str, Any]] = []

        # Get the full grid of segments (with addressing info)
        grid = self.segment_calculator.calculate_segment_grid(segments, h, w, image)

        for segment in grid:
            segment_data = segment['data']
            if segment_data.size > 0:
                segment_entropy = self.analyze(segment_data)
                # Append entropy with positional information
                entropies.append({
                    'entropy': segment_entropy,
                    'start_i': segment['start_i'],
                    'start_j': segment['start_j'],
                    'end_i': segment['end_i'],
                    'end_j': segment['end_j']
                })

        return entropies


class FolderManager(ReportManager):
    """Handles creating the output folder structure based on the input folder structure, 
       and includes the analysis type in the output folder name.
    """
    
    def __init__(self, config: Dict[str, Any], input_file: Path, input_folder: Path, analysis_type: str):
        self.analysis_type = analysis_type
        self.input_file = input_file
        self.input_folder = input_folder

        # Create the output folder as a "sister" folder to the input folder
        parent_folder = input_folder.parent
        output_folder_name = f"{analysis_type}_results_{input_folder.name}"
        output_folder = parent_folder / output_folder_name

        super().__init__(output_folder)  # Initialize the ReportManager with the output path

    def create_output_folders(self) -> Path:
        """
        Creates output folders by mirroring the input folder structure and adding the analysis type.
        
        For example, if the input folder is C:/data/samples/sample_1, the output folder will be C:/data/samples/{analysis_type}_results_sample_1.
        """
        # Get the relative path of the input file within the input folder
        relative_path = self.input_file.relative_to(self.input_folder)

        # Create the output folder structure, mirroring the input folder structure
        output_path = self.output_subfolder / relative_path.parent
        output_path.mkdir(parents=True, exist_ok=True)

        return output_path


class StatisticsManager(ReportManager):
    def __init__(self, output_subfolder: Path):
        super().__init__(output_subfolder)
        self.all_results: List[Tuple[int, float, float, float, float, np.ndarray]] = []

    def update_results(
        self,
        segments: int,
        avg_entropy: float,
        std_dev: float,
        min_entropy: float,
        max_entropy: float,
        entropies: List[Dict[str, Any]],
    ) -> None:
        # # Validate the structure of entropies
        if not isinstance(entropies, list) or not all(isinstance(entry, dict) and 'entropy' in entry for entry in entropies):
            print("Invalid entropies structure passed to update_results.")
            return  # Early exit to prevent further errors

        # Extract entropy values for storing
        entropy_values = [entry['entropy'] for entry in entropies]
        self.all_results.append(
            (segments, avg_entropy, std_dev, min_entropy, max_entropy, np.array(entropy_values))
        )

    def calculate_statistics(
        self, entropies: List[Dict[str, Any]]  # Expecting list of dictionaries
    ) -> Tuple[float, float, float, float]:
        if not entropies:
            return 0.0, 0.0, 0.0, 0.0  # Handle empty case

        # Extract entropy values from the list of dictionaries
        entropy_values = [entry['entropy'] for entry in entropies]
        entropies_statistics: Tuple[float, float, float, float] = (
            np.mean(entropy_values),
            np.std(entropy_values),
            np.min(entropy_values),
            np.max(entropy_values),
        )
        return entropies_statistics


    def clear_results(self) -> None:
        self.all_results.clear()

    def save_statistics_report(self, report_filename: str) -> None:
        logger.info(f"Saving statistics report to: {report_filename}")
        try:
            with open(report_filename, "w") as file:
                file.write("Segments, Average Entropy, Std Dev, Min Entropy, Max Entropy\n")
                for result in self.all_results:
                    file.write(
                        f"{result[0]}, {result[1]:.2f}, {result[2]:.2f}, {result[3]:.2f}, {result[4]:.2f}\n"
                    )
        except FileNotFoundError as e:
            logger.error(f"Failed to save report: {e}")
            raise


class PlotReportGenerator(ReportManager):
    def plot_entropy_variation(
        self,
        segments: int,
        entropies: List[Dict[str, Any]],
        config_manager: SSEAConfigurationManager,
        segment_calculator: SegmentCalculator,
        image_shape: Tuple[int, int],
        filename: str,
        num_rows: int,
        num_cols: int,
        output_file_path: Path
    ) -> None:
        self.ensure_output_folder()

        heatmap_data = np.zeros(image_shape)
        for entry in entropies:
            heatmap_data[entry['start_i']:entry['end_i'], entry['start_j']:entry['end_j']] = entry['entropy']

        height, width = image_shape
        aspect_ratio = width / height
        scale = 5
        figsize = (scale * aspect_ratio, scale)

        fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(
            heatmap_data,
            cmap="hot",
            aspect="equal",  # Critical: maintain pixel ratio
            interpolation="none",
            vmin=config_manager.segmentation_vmin,
            vmax=config_manager.segmentation_vmax
        )

        ax.set_xticks([])
        ax.set_yticks([])

        # Use divider to match colorbar height to image
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=config_manager.segmentation_cbar_text_size)

        # Add a title with spacing
        ax.set_title(
            f"Heatmap for {filename} - Segments: {segments}",
            fontsize=config_manager.segmentation_cbar_text_size,
            pad=20  # Adds spacing between title and image
        )

        heatmap_file_path = output_file_path / f"{filename}_heatmap_{segments}.tif"
        plt.savefig(
            heatmap_file_path,
            format="tiff",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.3,
            pil_kwargs={"compression": "tiff_adobe_deflate"}
        )
        plt.close(fig)

    def save_raw_entropy_data(
        self, report_filename: str, entropies: List[Dict[str, Any]], num_rows: int, num_cols: int, 
        segment_size: int, segmentation_method: str, h: int, w: int
    ) -> None:
        logger.info(f"Saving raw entropy data to: {report_filename}")
        try:
            with open(report_filename, "w") as file:
                file.write(f"# Segmentation Method: {segmentation_method}\n")
                file.write(f"# Segment Dimensions (Square Size): {segment_size}x{segment_size}\n")
                file.write(f"# Image Size: {h}x{w}\n")
                file.write("Start_i,Start_j,End_i,End_j,Entropy\n")

                for entry in entropies:
                    start_i = entry['start_i']
                    start_j = entry['start_j']
                    end_i = entry['end_i']
                    end_j = entry['end_j']
                    entropy_value = entry['entropy']

                    # Write pixel coordinates with entropy values
                    file.write(f"{start_i},{start_j},{end_i},{end_j},{entropy_value:.6f}\n")
        except FileNotFoundError as e:
            logger.error(f"Failed to save raw entropy data: {e}")
            raise


class SegmentedShannonEntropyAnalysisController:
    def __init__(self, config: Dict[str, Any], input_file: Path, input_folder: Path, analysis_type: str):
        # Pass analysis_type to SSEAConfigurationManager, which inherits BaseConfigurationManager
        self.config_manager = SSEAConfigurationManager(config, input_file, input_folder, analysis_type)

        # The rest of the setup
        self.segment_calculator = self._select_segment_calculator(self.config_manager.segmentation_method)
        self.entropy_analyzer = EntropyAnalyzer(self.segment_calculator)

        # Setup statistics manager and visualizations (use the output folder from config_manager)
        self.statistics_manager = StatisticsManager(self.config_manager.output_folder)
        self.visualization_manager = PlotReportGenerator(self.config_manager.output_folder)

    def _select_segment_calculator(self, method: str) -> SegmentCalculator:
        """Selects the appropriate segment calculator based on the method provided."""
        if method == "crop":
            return CropSegmentCalculator()
        if method == "merge":
            return MergeSegmentCalculator()
        if method == "adaptive":
            return AdaptiveSegmentCalculator()
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

    def run_analysis(self, file_path: Path) -> None:
        """
        Standardized run method for compatibility with entry.py.
        """
        self.run_segmented_analysis(file_path, self.config_manager.input_folder)
        
    def run_segmented_analysis(self, image_path: Path, image_folder) -> None:
        """Oversees the full analysis process for a single image."""

        # Step 1: Convert the image to a format ready for analysis
        self._prepare_image(image_path)

        # Step 2: Analyze the image for different segmentations
        self._analyze_image()

        # Step 3: Save the results (both visual and textual)
        self._save_results(image_path, image_folder)

    def _prepare_image(self, image_path: Path) -> None:
        """Prepares the image for analysis, including loading and converting."""
        self.config_manager.filename = image_path.name

        supported_image_extensions = [".png", ".tif", ".jpg", ".jpeg"]
        if image_path.suffix.lower() in supported_image_extensions:
            image_converter = ImageConverter(image_path)
            self.config_manager.image = image_converter.convert_to_npy()
            self.config_manager.bit_depth = image_converter.bit_depth
            self.config_manager.segmentation_vmax = image_converter.bit_depth
        else:
            self.config_manager.image = np.load(image_path)
            self.config_manager.bit_depth = self.config_manager.image.dtype.itemsize * 8
            self.config_manager.segmentation_vmax = self.config_manager.bit_depth

    def _analyze_image(self) -> None:
        """Performs entropy analysis on the image for different segmentation numbers."""
        h, w = self.config_manager.image.shape[:2]
        self.statistics_manager.clear_results()  # Clear previous results

        for segments in self.config_manager.segmentation_number:
            # Get the list of entropy results with positional info
            entropy_data = self.entropy_analyzer.calculate_segment_entropies(
                self.config_manager.image, segments, h, w
            )
            
            # You can calculate the num_rows and num_cols here based on entropy_data
            if not entropy_data:
                logger.warning("No entropy data returned.")
                continue  # Skip this segment if there's no data
            
            num_rows = max(entry['start_i'] for entry in entropy_data) // (entropy_data[0]['end_i'] - entropy_data[0]['start_i']) + 1
            num_cols = max(entry['start_j'] for entry in entropy_data) // (entropy_data[0]['end_j'] - entropy_data[0]['start_j']) + 1

            # Calculate statistics and update results
            stats = self.statistics_manager.calculate_statistics(entropy_data)  # Use the original structure
            self.statistics_manager.update_results(segments, *stats, entropy_data)  # Pass the original structure

    def _save_results(self, image_path: Path, input_folder: Path) -> None:
        """Saves the entropy analysis results, including heatmaps and raw data, preserving the input folder structure in the output folder."""
        
        # Get the relative path of the input file within the input folder
        relative_path = image_path.relative_to(input_folder)

        # Construct the output folder path without duplication
        output_file_path = self.config_manager.output_folder / relative_path.parent
        output_file_path.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        # Define the path for the statistics file
        statistics_file = output_file_path / f"{image_path.stem}_statistics.csv"

        for segments in self.config_manager.segmentation_number:
            # Get the full data from calculate_segment_entropies
            entropy_data = self.entropy_analyzer.calculate_segment_entropies(
                self.config_manager.image, segments, 
                self.config_manager.image.shape[0], self.config_manager.image.shape[1]
            )

            # Ensure you pass the original entropy_data (list of dicts) to the plot
            num_rows = max(entry['start_i'] for entry in entropy_data) // (entropy_data[0]['end_i'] - entropy_data[0]['start_i']) + 1
            num_cols = max(entry['start_j'] for entry in entropy_data) // (entropy_data[0]['end_j'] - entropy_data[0]['start_j']) + 1

            # Save the heatmap using the correct path
            if self.config_manager.generate_plots:
                self.visualization_manager.plot_entropy_variation(
                    segments,
                    entropy_data,
                    self.config_manager,
                    self.segment_calculator,
                    self.config_manager.image.shape[:2],
                    image_path.stem,
                    num_rows,
                    num_cols,
                    output_file_path
                )

            # Save the raw entropy data with the segmentation number in the filename
            raw_entropy_file = output_file_path / f"{image_path.stem}_raw_entropy_data_{segments}.csv"
            segment_size = segments
            self.visualization_manager.save_raw_entropy_data(
                raw_entropy_file, entropy_data, num_rows, num_cols, 
                segment_size, self.config_manager.segmentation_method,
                self.config_manager.image.shape[0], self.config_manager.image.shape[1]
            )
        
        # Save the statistics report after processing all segments
        self.statistics_manager.save_statistics_report(statistics_file)