# MiPAS – Modular Image Processing and Analysis Suite

MiPAS is a modular image processing and analysis toolkit designed for high-resolution microscopy data. It provides GUI- and batch-driven tools to perform various types of entropy and fractal analyses on grayscale or preprocessed image files.

---

## Features

- GUI with splash screen (`start_MiPAS.py`)
- Multiple analysis types:
  - Linear Shannon entropy
  - Segmented Shannon entropy
  - Fractal dimension (box-counting and WTMM)
- Modular configuration and multiprocessing support
- Structured outputs: plots, CSVs, logs
- Compatible with Python 3.10.5

---

## Installation

See [`INSTALLATION_GUIDE_MiPAS.md`](docs/INSTALLATION_GUIDE_MiPAS.md) for complete steps.

Python 3.10.5 is required.

---

## How to Run

### Launch the GUI

```bash
python mipas/start_MiPAS.py
```

Or double-click:

```
start_MiPAS.bat
```

### Run batch analysis programmatically

```python
from mipas.entry import run_analysis

run_analysis(config_dict, analysis_type="linear")
```

---

## Directory Structure

```
[.]
  └── CITATION.CFF
  └── folder_structure.py
  └── requirements.txt
  └── setup.py
  └── start_MiPAS.bat

[docs]
  └── CHANGELOG.md
  └── INSTALLATION_GUIDE_MiPAS.md
  └── license.txt
  └── README.md

[mipas]
  └── entry.py
  └── logging_config.py
  └── start_MiPAS.py
  └── __init__.py

[mipas\analysis]
  └── analysis_template.py
  └── fd_box_analysis.py
  └── fd_wtmm_analysis.py
  └── linear_entropy_analysis.py
  └── segmented_entropy_analysis.py
  └── __init__.py

[mipas\config]
  └── configuration_manager.py
  └── config_bootstrap.py
  └── config_linear_entropy_analysis.py
  └── config_registry.py
  └── config_schema.py
  └── config_segmented_entropy_analysis.py
  └── __init__.py

[mipas\config\json_configs]
  └── fd_box_config.json
  └── fd_wtmm_config.json
  └── linear_config.json
  └── segmented_config.json

[mipas\gui]
  └── gui_config_editor.py
  └── gui_launcher.py
  └── gui_main.py
  └── __init__.py

[mipas\logo]
  └── mipas_logo.png
```

---

## Supported Input Formats

- Image formats: `.png`, `.tif`, `.jpg`, `.jpeg`
- Arrays: `.npy` (2D intensity arrays)
- Images must be grayscale or convertible to grayscale

---

## Output

- Entropy plots (`.tif`)
- Raw CSVs: entropy values, stats
- Logs for batch or GUI execution

---

## Requirements

- Python 3.10.5
- Libraries:
  - Pillow
  - matplotlib
  - numpy
  - scipy
  - scikit-image
  - PySide2

---

## License

See [`docs/license.txt`]

---

## Changelog

See [`docs/CHANGELOG.md`]
