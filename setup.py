from setuptools import setup, find_packages

setup(
    name="mipas",
    version="1.0.2",
    packages=find_packages(include=["mipas", "mipas.*"]),
    install_requires=[
        "numpy",
        "scipy",
        "Pillow",
        "matplotlib",
        "scikit-image"
    ],
    python_requires="=3.10.5",
    entry_points={
        "console_scripts": [
            "mipas-gui = mipas.gui.gui_launcher:main",
        ]
    },
    include_package_data=True,
)
