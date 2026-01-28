from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout


class StatusPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)

        self.person = QLabel()
        self.first = QLabel()
        self.scans = QLabel()
        self.verdict = QLabel()

        for w in [self.person, self.first, self.scans, self.verdict]:
            layout.addWidget(w)

    def update(self, meta, emotions):
        person_id, first_emotion, verdict = meta

        self.person.setText(f"Person: {person_id}")
        self.first.setText(f"First Emotion: {first_emotion or '-'}")
        self.scans.setText(f"Scans: {len(emotions)} / 10")

        if verdict:
            self.verdict.setText(verdict)
            if "High Risk" in verdict:
                self.verdict.setStyleSheet("color: red; font-weight: bold")
            elif "Distressed" in verdict:
                self.verdict.setStyleSheet("color: orange")
            else:
                self.verdict.setStyleSheet("color: green")
        else:
            self.verdict.setText("Scanning in progress")
            self.verdict.setStyleSheet("color: gray")
