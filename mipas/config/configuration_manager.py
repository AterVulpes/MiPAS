# mipas/config/configuration_manager.py

from pathlib import Path
from mipas.config.config_schema import ConfigSchema
import logging

logger = logging.getLogger(__name__)

class ConfigurationManager:
    def __init__(self, config: dict, input_file: Path, input_folder: Path, analysis_type: str):
        self.analysis_type = analysis_type
        self.filename = input_file.name
        self.input_folder = input_folder

        # Validate and store config via central manager
        self.cfg = ConfigSchema(config, analysis_type)

        # Set output folder
        self.output_folder = Path(
            config.get("output_folder") or
            input_folder.parent / f"{analysis_type}_results_{input_folder.name}"
        )
        self._validate_and_create_folder(self.output_folder)

        self._log_config()

    def _validate_and_create_folder(self, folder_path: Path):
        if not folder_path.exists():
            try:
                folder_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created output folder: {folder_path}")
            except Exception as e:
                logger.exception(f"Failed to create output folder: {folder_path}")
                raise FileNotFoundError(f"Cannot create output folder: {folder_path}")

    def _log_config(self):
        logger.info(f"Configuration for {self.analysis_type} analysis:")
        for k, v in self.cfg.to_dict().items():
            logger.info(f"  {k}: {v}")
