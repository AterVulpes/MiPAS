# mipas/config/config_linear_entropy_analysis.py

CONFIG = {
    "input_folder": {
        "type": "path",
        "default": "",
        "label": "Input Folder",
        "hint": "Select the folder containing the input images for analysis.",
        "validate": lambda x: isinstance(x, str),
    },
    "sliding_window_entropy_percent": {
        "type": "float",
        "default": 10.0,
        "label": "sliding Window Size (%)",
        "hint": "Specify the size of the sliding window as a percentage of the image height. "
                "The sliding window is used to compute localized entropy variations. "
                "Recommended range: 5–20%. Value must be between 1 and 99.",
        "validate": lambda x: isinstance(x, (float, int)) and 1 <= x <= 99,
    },
    "title_size": {
        "type": "int",
        "default": 24,
        "label": "Plot Title Font Size",
        "hint": "Font size for the plot title in points (e.g., 24).",
        "validate": lambda x: isinstance(x, int) and x > 0,
    },
    "axis_label_size": {
        "type": "int",
        "default": 22,
        "label": "Axis Label Font Size",
        "hint": "Font size for X and Y axis labels in points (e.g., 22).",
        "validate": lambda x: isinstance(x, int) and x > 0,
    },
    "tick_label_size": {
        "type": "int",
        "default": 20,
        "label": "Tick Label Font Size",
        "hint": "Font size for tick labels on axes (e.g., 20).",
        "validate": lambda x: isinstance(x, int) and x > 0,
    },
    "legend_size": {
        "type": "int",
        "default": 18,
        "label": "Legend Font Size",
        "hint": "Font size for the legend text (e.g., 18).",
        "validate": lambda x: isinstance(x, int) and x > 0,
    },
    "dpi_setting": {
        "type": "int",
        "default": 300,
        "label": "Plot DPI Setting",
        "hint": "Resolution of the saved plot in dots per inch (e.g., 300 for high quality printing).",
        "validate": lambda x: isinstance(x, int) and x >= 72,
    },
    "plot_figsize": {
        "type": "list",
        "default": [16, 8],
        "label": "Plot Figure Size",
        "hint": "Size of the figure in inches as [width, height] (e.g., [16, 8]).",
        "validate": lambda x: isinstance(x, (list, tuple)) and len(x) == 2 and all(isinstance(i, (int, float)) and i > 0 for i in x),
    },
    "max_retries": {
        "type": "int",
        "default": 3,
        "label": "Max Retries on Failure",
        "hint": "Maximum number of times to retry processing a file if errors occur (default 3).",
        "validate": lambda x: isinstance(x, int) and x >= 0,
    },
    "plot": {
        "type": "bool",
        "default": True,
        "label": "Generate Plots",
        "hint": "If checked, visual figures will be generated and saved. Unchecked = CSV only.",
        "validate": lambda x: isinstance(x, bool),
    }
}
