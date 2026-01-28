from PySide6.QtWidgets import (
    QMainWindow, QWidget, QListWidget,
    QVBoxLayout, QHBoxLayout
)
from PySide6.QtCore import QTimer

from mental_state_module.storage.person_store import (
    get_all_persons, get_emotions
)
from mental_state_module.charts.charts_widget import ChartsWidget
from mental_state_module.gui.status_panel import StatusPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mental State Dashboard")
        self.resize(1200, 700)

        root = QWidget()
        self.setCentralWidget(root)

        layout = QHBoxLayout(root)

        self.person_list = QListWidget()
        self.person_list.currentTextChanged.connect(self.load_person)
        layout.addWidget(self.person_list, 2)

        right = QVBoxLayout()
        layout.addLayout(right, 8)

        self.status_panel = StatusPanel()
        self.charts = ChartsWidget()

        right.addWidget(self.status_panel)
        right.addWidget(self.charts)

        self.refresh_persons()

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh_current)
        self.timer.start(2000)

    def refresh_persons(self):
        self.person_list.clear()
        for p in get_all_persons():
            self.person_list.addItem(p[0])

    def load_person(self, person_id):
        self.current_person = person_id
        self.refresh_current()

    def refresh_current(self):
        if not hasattr(self, "current_person"):
            return

        emotions = get_emotions(self.current_person)
        persons = get_all_persons()
        meta = next(p for p in persons if p[0] == self.current_person)

        self.status_panel.update(meta, emotions)
        self.charts.update(emotions)
