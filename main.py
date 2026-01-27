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
