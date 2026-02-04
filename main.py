import os
import sys
import io

# ------------------------------
# FIX Windows stdout crash
# ------------------------------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="ignore")

# ------------------------------
# Silence TensorFlow INFO logs
# ------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import sys
from PySide6.QtWidgets import QApplication
from main_window import MainWindow


def main():
    """
    ENTRY POINT
    This file MUST be run using the main project venv (venv)
    """
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
