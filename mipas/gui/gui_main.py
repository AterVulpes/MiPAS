# mipas/gui/gui_main.py

"""
MiPAS GUI Main (PySide2 version)

Provides the main GUI for the Modular Image Processing and Analysis Suite (MiPAS).
"""

# ============ Standard Library Imports ============
import sys
import json
import os
from pathlib import Path

# ============ Third-party Imports ============
from PySide2.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QCheckBox, QTextEdit, QTabWidget, QScrollArea, QFrame, QGroupBox
)
from PySide2.QtCore import Qt

# ============ Project-specific Imports ============
from mipas.config.config_bootstrap import initialize_config_file_if_missing
from mipas.gui.gui_config_editor import open_config_editor_window
from mipas.logging_config import setup_main_logger
from mipas.entry import run_analysis

# ============ Logging setup ============
logger = setup_main_logger()

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config" / "json_configs"

ANALYSIS_TYPES = {
    "linear": {
        "label": "Linear Entropy",
        "config_file": CONFIG_DIR / "linear_config.json",
        "status": "stable"
    },
    "segmented": {
        "label": "Segmented Entropy",
        "config_file": CONFIG_DIR / "segmented_config.json",
        "status": "stable"
    },
    "fd-box": {
        "label": "Fractal Dimension (Box Counting)",
        "config_file": CONFIG_DIR / "fd_box_config.json",
        "status": "experimental"
    },
    "fd-wtmm": {
        "label": "Fractal Dimension (WTMM)",
        "config_file": CONFIG_DIR / "fd_wtmm_config.json",
        "status": "experimental"
    }
}

for key, meta in ANALYSIS_TYPES.items():
    initialize_config_file_if_missing(key, meta["config_file"])

class MiPASLauncher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MiPAS Analysis Launcher")
        self.resize(800, 600)

        self.state = {
            "input_folder": "",
            "selected_analyses": {},
            "configs": {},
            "folder_overrides": {},
            "status_log": None
        }

        self.load_configs()
        self.build_ui()

    def load_configs(self):
        for key, meta in ANALYSIS_TYPES.items():
            try:
                with open(meta["config_file"], "r") as f:
                    config = json.load(f)
                self.state["configs"][key] = config
                logger.info(f"Loaded config for '{key}' from {meta['config_file']}")
            except Exception as e:
                logger.error(f"Failed to load config file '{meta['config_file']}': {e}")
                self.state["configs"][key] = {}

            self.state["selected_analyses"][key] = False

    def build_ui(self):
        layout = QVBoxLayout(self)

        # Folder Selector
        folder_layout = QHBoxLayout()
        folder_label = QLabel("Input Folder:")
        self.folder_path_label = QLabel("(None)")
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_folder)

        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self.folder_path_label, 1)
        folder_layout.addWidget(browse_button)

        layout.addLayout(folder_layout)

        # Tabs
        self.tabs = QTabWidget()
        self.stable_tab = QWidget()
        self.experimental_tab = QWidget()

        self.tabs.addTab(self.stable_tab, "Analysis Methods")
        self.tabs.addTab(self.experimental_tab, "Experimental Support")

        self.build_analysis_section(self.stable_tab, experimental_only=False)
        self.build_analysis_section(self.experimental_tab, experimental_only=True)

        layout.addWidget(self.tabs)

        # Run button
        run_button = QPushButton("Run Analyses")
        run_button.clicked.connect(self.run_selected_analyses)
        layout.addWidget(run_button)

        # Status log
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        layout.addWidget(self.status_log, 1)

        self.update_folder_indicators()

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder and os.path.isdir(folder):
            self.state["input_folder"] = folder
            self.folder_path_label.setText(folder)
            self.update_folder_indicators()
            logger.info(f"Input folder selected: {folder}")
        else:
            logger.warning("Invalid folder selected.")

    def build_analysis_section(self, tab, experimental_only):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content = QWidget()
        vbox = QVBoxLayout(content)

        for key, meta in ANALYSIS_TYPES.items():
            is_experimental = meta.get("status", "stable") == "experimental"
            if is_experimental != experimental_only:
                continue

            group = QGroupBox(meta["label"])
            group_layout = QHBoxLayout(group)

            checkbox = QCheckBox()
            checkbox.setChecked(self.state["selected_analyses"].get(key, False))
            checkbox.stateChanged.connect(lambda state, k=key: self.toggle_analysis(k, state))
            group_layout.addWidget(checkbox)

            edit_button = QPushButton("Edit Config")
            edit_button.clicked.connect(lambda checked=False, k=key: self.open_config_editor(k))
            group_layout.addWidget(edit_button)

            override_label = QLabel("")
            group_layout.addWidget(override_label)
            self.state["folder_overrides"][key] = override_label

            vbox.addWidget(group)

        vbox.addStretch(1)
        scroll_area.setWidget(content)
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(scroll_area)

    def toggle_analysis(self, key, state):
        self.state["selected_analyses"][key] = (state == Qt.Checked)

    def open_config_editor(self, key):
        open_config_editor_window(self, key, self.state["configs"][key], self.update_folder_indicators)

    def update_folder_indicators(self):
        for key, label in self.state["folder_overrides"].items():
            cfg = self.state["configs"].get(key)
            if not cfg:
                continue
            path = cfg.get("input_folder", "")
            label.setText("(custom)" if path else "(using global)")

    def run_selected_analyses(self):
        selected = [key for key, val in self.state["selected_analyses"].items() if val]
        if not selected:
            self.status_log.append("Please select at least one analysis type.")
            logger.warning("No analysis selected.")
            return

        folder = self.state["input_folder"]
        missing_folder = False

        for key in selected:
            config = self.state["configs"][key]
            if config.get("input_folder") and os.path.isdir(config["input_folder"]):
                continue
            if folder and os.path.isdir(folder):
                config["input_folder"] = folder
            else:
                missing_folder = True
                break

        if missing_folder:
            self.status_log.append("Please select a valid input folder.")
            logger.warning("Missing or invalid input folder.")
            return

        self.status_log.clear()
        self.status_log.append("Starting analyses...")

        for key in selected:
            config = self.state["configs"][key]
            try:
                run_analysis(config=config, analysis_type=key, num_workers=-1)
                self.status_log.append(f"{key} completed.")
                logger.info(f"{key} analysis completed.")
            except Exception as e:
                self.status_log.append(f"{key} failed: {e}")
                logger.exception(f"{key} failed.")

        self.status_log.append("All selected analyses finished.")

def launch_gui():
    app = QApplication(sys.argv)
    window = MiPASLauncher()
    window.show()
    sys.exit(app.exec_())
