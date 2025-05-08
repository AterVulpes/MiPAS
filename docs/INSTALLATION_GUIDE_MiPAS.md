# MiPAS Installation Guide

This guide will help you install and run MiPAS, even if you're new to Python.

## 1. Install Python 3.10.5 (Windows)

1. Download Python 3.10.5 from the [official archives](https://www.python.org/downloads/release/python-3105/).
2. Run the installer.
3. **IMPORTANT:** Check the box that says **“Add Python to PATH”** before clicking **Install Now**.
4. Verify the installation:
   - Open **Command Prompt** (`Win + R`, type `cmd`, press Enter).
   - Type:
     ```cmd
     python --version
     ```
   - You should see:
     ```
     Python 3.10.5
     ```

## 2. Install Required Libraries

Once Python is installed, open Command Prompt and install the required libraries.

### Option A: Install from `requirements.txt` (if provided)

Navigate to the MiPAS project folder and run:

```cmd
cd "L:\your\MiPAS\project\folder"
pip install -r requirements.txt
```

### Option B: Manual install (if `requirements.txt` is not available)

Run the following commands one by one:

```cmd
pip install Pillow
pip install matplotlib
pip install numpy
pip install scipy
pip install scikit-image
pip install PySide2
```

## 3. Launch MiPAS

### Option A: Use the batch launcher

Double-click the file:

```
start_MiPAS.bat
```

### Option B: Launch from the command line

Navigate to the project folder and run:

```cmd
python mipas/start_MiPAS.py
```

## Troubleshooting

- If `python` is not recognized, restart Command Prompt or re-install Python ensuring **“Add to PATH”** is checked.
- If `pip` fails, try running as administrator.
