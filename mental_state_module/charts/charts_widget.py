from collections import Counter

from PySide6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class ChartsWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)

        layout.addWidget(self.canvas)

    def update(self, emotions):
        """
        emotions = [(emotion, confidence, timestamp), ...]
        """
        self.figure.clear()

        if not emotions:
            self.canvas.draw()
            return

        emotion_list = [e[0] for e in emotions]
        counts = Counter(emotion_list)

        ax = self.figure.add_subplot(111)
        ax.bar(counts.keys(), counts.values())

        ax.set_title("Emotion Distribution")
        ax.set_xlabel("Emotion")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=30)

        self.canvas.draw()
