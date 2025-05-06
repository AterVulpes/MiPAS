# mipas/gui/gui_config_editor.py

from PySide2.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QScrollArea, QWidget, QMessageBox, QCheckBox
)
from PySide2.QtCore import Qt
from mipas.config.config_registry import CONFIG_REGISTRY


def open_config_editor_window(parent, analysis_key, config_dict, update_callback=None):
    print("SCHEMAS AVAILABLE:", list(CONFIG_REGISTRY.keys()))
    normalized_key = {
        "linear": "linear_entropy_analysis",
        "segmented": "segmented_entropy_analysis",
        "fd-box": "fd_box_analysis",
        "fd-wtmm": "fd_wtmm_analysis"
    }.get(analysis_key, analysis_key)

    schema = CONFIG_REGISTRY.get(normalized_key, {})

    dialog = QDialog(parent)
    dialog.setWindowTitle(f"Edit Config: {analysis_key}")
    dialog.resize(700, 500)
    dialog.setModal(True)

    main_layout = QVBoxLayout(dialog)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll_content = QWidget()
    scroll_layout = QVBoxLayout(scroll_content)

    editor_state = {}

    if not schema:
        hint_label = QLabel("(No schema available, basic editing only)")
        hint_label.setStyleSheet("color: blue; font-style: italic;")
        scroll_layout.addWidget(hint_label)

    def add_config_row(key, value):
        field_meta = schema.get(key, {})
        field_type = field_meta.get("type", "str")
        label_text = field_meta.get("label", key)
        hint_text = field_meta.get("hint", "")
        editable = field_meta.get("editable", True)

        row_container = QWidget()
        row_layout = QVBoxLayout(row_container)

        field_row = QWidget()
        field_row_layout = QHBoxLayout(field_row)
        field_row_layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(label_text)
        label.setFixedWidth(200)
        field_row_layout.addWidget(label)

        if field_type == "bool":
            entry = QCheckBox()
            entry.setChecked(bool(value))
            field_row_layout.addWidget(entry)
        elif field_type == "path" or key == "input_folder":
            entry = QLineEdit(str(value))
            entry.setReadOnly(not editable)
            browse_button = QPushButton("Browse")

            def browse():
                folder = QFileDialog.getExistingDirectory(dialog, "Select Folder")
                if folder:
                    entry.setText(folder)

            browse_button.clicked.connect(browse)
            field_row_layout.addWidget(entry, 1)
            field_row_layout.addWidget(browse_button)
        else:
            entry = QLineEdit(str(value) if value is not None else "")
            entry.setReadOnly(not editable)
            field_row_layout.addWidget(entry, 1)

        editor_state[key] = entry
        row_layout.addWidget(field_row)

        if hint_text:
            hint_label = QLabel(hint_text)
            hint_label.setStyleSheet("color: gray; font-size: 10px; margin-left: 5px;")
            hint_label.setWordWrap(True)
            row_layout.addWidget(hint_label)

        scroll_layout.addWidget(row_container)

    for key, meta in schema.items():
        value = config_dict.get(key, meta.get("default", ""))
        add_config_row(key, value)

    scroll_content.setLayout(scroll_layout)
    scroll.setWidget(scroll_content)
    main_layout.addWidget(scroll)

    # Save/Cancel buttons
    button_layout = QHBoxLayout()
    save_button = QPushButton("Save")
    cancel_button = QPushButton("Cancel")
    button_layout.addWidget(save_button)
    button_layout.addWidget(cancel_button)
    main_layout.addLayout(button_layout)

    def on_save():
        for key, entry in editor_state.items():
            field_type = schema.get(key, {}).get("type", "str")

            try:
                if isinstance(entry, QCheckBox):
                    config_dict[key] = entry.isChecked()
                else:
                    val = entry.text().strip()
                    if field_type == "int":
                        config_dict[key] = int(val)
                    elif field_type == "float":
                        config_dict[key] = float(val)
                    elif field_type == "list":
                        config_dict[key] = eval(val) if val.startswith("[") else [int(x.strip()) for x in val.split(",")]
                    elif field_type in ("str", "path"):
                        config_dict[key] = val
                    else:
                        config_dict[key] = val
            except Exception as e:
                QMessageBox.critical(dialog, "Invalid Input", f"Invalid input for '{key}': {str(e)}")
                return

        if update_callback:
            update_callback()
        dialog.accept()

    def on_cancel():
        if QMessageBox.question(dialog, "Discard changes?", "Discard all changes?", QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            dialog.reject()

    save_button.clicked.connect(on_save)
    cancel_button.clicked.connect(on_cancel)

    dialog.exec_()
