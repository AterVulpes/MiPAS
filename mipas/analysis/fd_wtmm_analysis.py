# mipas/analysis/fd_wtmm_analysis.py

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import csv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage

from mipas.config.configuration_manager import ConfigurationManager

logger = logging.getLogger(__name__)


def compute_wavelet_transform_2d(image: np.ndarray, scales: list, smoothing_func: str = "gaussian") -> dict:
    results = {}
    for scale in scales:
        Tψ1 = scipy.ndimage.gaussian_filter(image, sigma=scale, order=[1, 0])
        Tψ2 = scipy.ndimage.gaussian_filter(image, sigma=scale, order=[0, 1])
        modulus = np.sqrt(Tψ1 ** 2 + Tψ2 ** 2)
        argument = np.arctan2(Tψ2, Tψ1)
        results[scale] = (modulus, argument, Tψ1, Tψ2)
    return results


class ImageConverter:
    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.bit_depth = None

    def convert_to_npy(self) -> np.ndarray:
        try:
            img = Image.open(self.image_path)
            if img.mode not in ["L", "I;16"]:
                img = img.convert("L")
            img_array = np.array(img)
            self.bit_depth = img_array.dtype.itemsize * 8
            return img_array
        except Exception as e:
            logger.error(f"Error converting image: {e}")
            raise


class WTMMFDConfigurationManager(ConfigurationManager):
    def __init__(self, config: Dict[str, Any], input_file: Path, input_folder: Path, analysis_type: str):
        super().__init__(config, input_file, input_folder, analysis_type)

        self.scales = self.cfg["scales"]
        self.q_values = self.cfg["q_values"]
        self.smoothing_func = self.cfg["smoothing_func"]

        converter = ImageConverter(input_file)
        self.image = converter.convert_to_npy()
        self.bit_depth = converter.bit_depth
        self.cfg.data["bit_depth"] = self.bit_depth  # for plots if needed

        logger.info(f"Loaded {input_file.name} with bit depth {self.bit_depth}")


class WTMMFractalCalculator:
    def __init__(self, image: np.ndarray, config: WTMMFDConfigurationManager):
        self.image = image.astype(np.float32)
        self.config = config

    def calculate_fractal_dimension(self) -> Tuple[float, list, list, list, list]:
        from scipy.ndimage import maximum_filter

        wt = compute_wavelet_transform_2d(self.image, self.config.scales, self.config.smoothing_func)
        partition_fn = {q: [] for q in self.config.q_values}

        for scale in self.config.scales:
            modulus = wt[scale][0]
            local_max = maximum_filter(modulus, size=3) == modulus
            maxima_values = modulus[local_max]
            maxima_values = maxima_values[maxima_values > 0]

            for q in self.config.q_values:
                partition = np.sum(maxima_values ** q)
                partition_fn[q].append(partition)

        log_scales = np.log(self.config.scales)
        tau_q = []
        for q in self.config.q_values:
            log_partition = np.log(partition_fn[q])
            slope, _ = np.polyfit(log_scales, log_partition, 1)
            tau_q.append(slope)

        D = -np.interp(0, self.config.q_values, tau_q)
        log_N_q0 = np.log(partition_fn[0.0])
        return D, list(log_scales), list(log_N_q0), list(self.config.q_values), list(tau_q)


class WTMMTextReportGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def save_summary(self, relative_path: Path, fd_value: float):
        with open(self.output_dir / "fd_wtmm_summary.txt", "a", encoding="utf-8") as f:
            f.write(f"{relative_path.as_posix()}\t{fd_value:.4f}\n")


class WTMMPlotReportGenerator:
    def __init__(self, output_dir: Path, config: WTMMFDConfigurationManager):
        self.output_dir = output_dir
        self.cfg = config.cfg

    def save_plot(self, log_r, log_N, fd_value: float, image_path: Path):
        plt.figure(figsize=self.cfg["plot_figsize"])
        plt.plot(log_r, log_N, 'o-', label=f"FD (WTMM): {fd_value:.4f}")
        slope, intercept = np.polyfit(log_r, log_N, 1)
        plt.plot(log_r, np.polyval([slope, intercept], log_r), 'r--', label=f"Slope: {slope:.4f}")
        plt.xlabel("log(scale)", fontsize=self.cfg["axis_label_size"])
        plt.ylabel("log(sum of maxima)", fontsize=self.cfg["axis_label_size"])
        plt.title(f"WTMM FD: {image_path.name}", fontsize=self.cfg["title_size"])
        plt.xticks(fontsize=self.cfg["tick_label_size"])
        plt.yticks(fontsize=self.cfg["tick_label_size"])
        plt.legend(fontsize=self.cfg["legend_size"])
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{image_path.stem}_fd_wtmm_plot.tif", dpi=self.cfg["dpi_setting"])
        plt.close()

    def save_tau_dh_plots(self, q_vals, tau_q, image_path: Path):
        q_vals = np.array(q_vals)
        tau_q = np.array(tau_q)
        dq = np.gradient(q_vals)
        dtau = np.gradient(tau_q, dq)
        h = dtau
        D_h = q_vals * h - tau_q

        # τ(q)
        plt.figure(figsize=self.cfg["plot_figsize"])
        plt.plot(q_vals, tau_q, 'bo-', label='τ(q)')
        plt.xlabel("q", fontsize=self.cfg["axis_label_size"])
        plt.ylabel("τ(q)", fontsize=self.cfg["axis_label_size"])
        plt.title(f"Multifractal Spectrum τ(q): {image_path.name}", fontsize=self.cfg["title_size"])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{image_path.stem}_tauq_plot.tif", dpi=self.cfg["dpi_setting"])
        plt.close()

        # D(h)
        plt.figure(figsize=self.cfg["plot_figsize"])
        plt.plot(h, D_h, 'go-', label='D(h)')
        plt.xlabel("h (Hölder exponent)", fontsize=self.cfg["axis_label_size"])
        plt.ylabel("D(h)", fontsize=self.cfg["axis_label_size"])
        plt.title(f"Multifractal Spectrum D(h): {image_path.name}", fontsize=self.cfg["title_size"])
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{image_path.stem}_dh_plot.tif", dpi=self.cfg["dpi_setting"])
        plt.close()

        # CSV exports
        with open(self.output_dir / f"{image_path.stem}_tauq_data.csv", "w", newline="") as f:
            csv.writer(f).writerows([["q", "tau(q)"]] + list(zip(q_vals, tau_q)))

        valid = np.isfinite(h) & np.isfinite(D_h)
        with open(self.output_dir / f"{image_path.stem}_dh_data.csv", "w", newline="") as f:
            csv.writer(f).writerows([["h", "D(h)"]] + list(zip(h[valid], D_h[valid])))


class WTMMFDAnalysisController:
    def __init__(self, config: Dict[str, Any], image_path: Path, input_folder: Path, analysis_type: str):
        self.config_manager = WTMMFDConfigurationManager(config, image_path, input_folder, analysis_type)

    def run_analysis(self, file_path: Path) -> None:
        self.run_wtmm_fd_analysis(file_path)

    def run_wtmm_fd_analysis(self, image_path: Path) -> None:
        try:
            fd_calc = WTMMFractalCalculator(self.config_manager.image, self.config_manager)
            fd_value, log_r, log_N, q_vals, tau_q = fd_calc.calculate_fractal_dimension()
            self._save_results(image_path, fd_value, log_r, log_N, q_vals, tau_q)
        except Exception as e:
            logger.error(f"Error in WTMM FD analysis for {image_path.name}: {str(e)}")
            raise

    def _save_results(self, image_path: Path, fd_value, log_r, log_N, q_vals, tau_q) -> None:
        output_dir = self.config_manager.output_folder / image_path.parent.name
        output_dir.mkdir(parents=True, exist_ok=True)

        relative_path = image_path.relative_to(self.config_manager.input_folder)

        WTMMTextReportGenerator(output_dir).save_summary(relative_path, fd_value)
        plotter = WTMMPlotReportGenerator(output_dir, self.config_manager)
        plotter.save_plot(log_r, log_N, fd_value, image_path)
        plotter.save_tau_dh_plots(q_vals, tau_q, image_path)
