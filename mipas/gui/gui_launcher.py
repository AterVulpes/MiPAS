#  project_MiPAS/gui/gui_launcher.py


"""
This module serves as the main entry point for launching the MiPAS graphical user interface (GUI).

It is responsible for:
    - Initializing the centralized logging system
    - Ensuring multiprocessing compatibility (especially on Windows)
    - Launching the Tkinter-based configuration and analysis GUI

This launcher is intended for use by end users who prefer an interactive, graphical workflow
rather than command-line interaction.

Usage:
    Run this module directly to launch the GUI:
        python gui_launcher.py

Modules Used:
    - multiprocessing: Ensures cross-platform compatibility (standard library)
    - mipas_logging: Centralized logging configuration (project)
    - gui_main: Main GUI construction and logic (project)
"""


# ================ Standard library imports ================
import sys
import multiprocessing


# ================ Project-specific Imports ================
from mipas.logging_config import setup_main_logger
from mipas.gui.gui_main import launch_gui


# ================ Main Execution ================
if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        # Context already set — fine if already configured earlier in the session
        pass

    # Initialize logging in the main process
    logger, log_listener = setup_main_logger()

    def main():
        try:
            logger.info("Starting the MiPAS application...")
            launch_gui()
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            sys.exit(1)
        except ImportError as e:
            logger.error(f"Import error: {e}")
            sys.exit(1)
        except RuntimeError as e:
            logger.error(f"Runtime error: {e}")
            sys.exit(1)
        except Exception:
            logger.exception("Unexpected error while launching the GUI")
            sys.exit(1)
        finally:
            try:
                logger.info("Shutting down MiPAS application.")
                log_listener.stop()
            except Exception as e:
                logger.warning(f"Logging listener shutdown error: {e}")

    main()
