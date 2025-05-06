# mipas/start_MiPAS.py

"""
Main user entry point to launch the MiPAS GUI from the top-level project folder.
"""

# ================ Standard library imports ================
import sys
import multiprocessing

# ================ Third-party imports ================
from PySide2.QtWidgets import QApplication, QLabel
from PySide2.QtGui import QPixmap
from PySide2.QtCore import Qt, QTimer

# ================ Project-specific Imports ================
from mipas.logging_config import setup_main_logger
from mipas.gui.gui_main import MiPASLauncher


def show_splash_screen(app, duration=2.0):
    splash = QLabel()
    splash.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SplashScreen)
    splash.setAttribute(Qt.WA_TranslucentBackground)

    pixmap = QPixmap("logo/mipas_logo.png")
    splash.setPixmap(pixmap)

    # Center the splash
    screen = app.primaryScreen().availableGeometry()
    x = (screen.width() - pixmap.width()) // 2
    y = (screen.height() - pixmap.height()) // 2
    splash.move(x, y)

    splash.show()
    app.processEvents()

    QTimer.singleShot(int(duration * 1000), splash.close)
    QTimer.singleShot(int(duration * 1000), app.quit)
    app.exec_()

def main():
    # Set up multiprocessing safety
    multiprocessing.set_start_method("spawn", force=True)

    # Initialize logging
    logger = setup_main_logger()
    logger.info("Starting MiPAS...")

    try:
        app = QApplication(sys.argv)  # <-- create app ONCE

        show_splash_screen(app, duration=2.0)  # <-- pass app in

        window = MiPASLauncher()  # <-- launch main window
        window.show()

        sys.exit(app.exec_())

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        try:
            logger.info("Shutting down MiPAS application.")
        except Exception as e:
            logger.warning(f"Logging shutdown error: {e}")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    main()
