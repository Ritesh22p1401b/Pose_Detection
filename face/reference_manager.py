import os
import shutil
from PySide6.QtWidgets import (
    QWidget, QListWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QMessageBox, QInputDialog
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, Signal

REFERENCE_DIR = "face/reference_faces"


class ReferenceManager(QWidget):
    closed = Signal()
    person_selected = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Reference Manager")
        self.setFixedSize(420, 550)

        os.makedirs(REFERENCE_DIR, exist_ok=True)

        # ---------------- LIST ----------------
        self.list = QListWidget()

        # 🔹 ENABLE MULTI-SELECTION (safe)
        self.list.setSelectionMode(QListWidget.ExtendedSelection)

        # ---------------- PREVIEW ----------------
        self.preview = QLabel("Image Preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedHeight(200)
        self.preview.setStyleSheet("border: 1px solid gray;")

        # ---------------- STATUS LABEL ----------------
        self.selection_label = QLabel("Selected persons: 0")
        self.selection_label.setAlignment(Qt.AlignCenter)
        self.selection_label.setStyleSheet("font-weight: bold;")

        # ---------------- BUTTONS ----------------
        self.add_person = QPushButton("Create Person")
        self.rename_person = QPushButton("Rename")
        self.delete_person = QPushButton("Delete")
        self.add_images = QPushButton("Add Images")

        # ---------------- LAYOUT ----------------
        layout = QVBoxLayout()
        layout.addWidget(self.list)
        layout.addWidget(self.preview)
        layout.addWidget(self.selection_label)

        btns = QHBoxLayout()
        btns.addWidget(self.add_person)
        btns.addWidget(self.rename_person)
        btns.addWidget(self.delete_person)

        layout.addLayout(btns)
        layout.addWidget(self.add_images)
        self.setLayout(layout)

        # ---------------- SIGNALS ----------------
        self.add_person.clicked.connect(self.create_person)
        self.rename_person.clicked.connect(self.rename)
        self.delete_person.clicked.connect(self.delete)
        self.add_images.clicked.connect(self.add_imgs)

        self.list.itemDoubleClicked.connect(self.select_person)
        self.list.itemSelectionChanged.connect(self.update_selection_count)
        self.list.currentItemChanged.connect(self.preview_image)

        self.refresh()
        self.update_selection_count()

    # --------------------------------------------------
    # WINDOW CLOSE
    # --------------------------------------------------
    def closeEvent(self, event):
        self.closed.emit()
        event.accept()

    # --------------------------------------------------
    # REFRESH LIST
    # --------------------------------------------------
    def refresh(self):
        self.list.clear()
        for name in sorted(os.listdir(REFERENCE_DIR)):
            if os.path.isdir(os.path.join(REFERENCE_DIR, name)):
                self.list.addItem(name)

    # --------------------------------------------------
    # SELECTION COUNT
    # --------------------------------------------------
    def update_selection_count(self):
        count = len(self.list.selectedItems())
        self.selection_label.setText(f"Selected persons: {count}")

    # --------------------------------------------------
    # DOUBLE-CLICK → LOAD SINGLE PERSON
    # --------------------------------------------------
    def select_person(self, item):
        self.person_selected.emit(item.text())

    # --------------------------------------------------
    # CREATE PERSON
    # --------------------------------------------------
    def create_person(self):
        name, ok = QInputDialog.getText(self, "Create Person", "Person name:")
        if not ok or not name.strip():
            return

        path = os.path.join(REFERENCE_DIR, name.strip())
        if os.path.exists(path):
            QMessageBox.warning(self, "Error", "Person already exists")
            return

        os.makedirs(path)
        self.refresh()

    # --------------------------------------------------
    # RENAME PERSON
    # --------------------------------------------------
    def rename(self):
        item = self.list.currentItem()
        if not item:
            QMessageBox.warning(self, "Error", "Select a person first")
            return

        old = item.text()
        new, ok = QInputDialog.getText(self, "Rename Person", "New name:")
        if ok and new.strip():
            os.rename(
                os.path.join(REFERENCE_DIR, old),
                os.path.join(REFERENCE_DIR, new.strip()),
            )
            self.refresh()

    # --------------------------------------------------
    # DELETE PERSON
    # --------------------------------------------------
    def delete(self):
        items = self.list.selectedItems()
        if not items:
            QMessageBox.warning(self, "Error", "Select at least one person")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete {len(items)} selected person(s)?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        for item in items:
            shutil.rmtree(os.path.join(REFERENCE_DIR, item.text()))

        self.refresh()
        self.preview.clear()
        self.update_selection_count()

    # --------------------------------------------------
    # ADD IMAGES
    # --------------------------------------------------
    def add_imgs(self):
        item = self.list.currentItem()
        if not item:
            QMessageBox.warning(self, "Error", "Select a person first")
            return

        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Images", "", "Images (*.jpg *.png)"
        )
        if not files:
            return

        for f in files:
            shutil.copy(f, os.path.join(REFERENCE_DIR, item.text()))

        QMessageBox.information(
            self, "Success",
            f"{len(files)} images added to {item.text()}"
        )

        self.preview_image(item)

    # --------------------------------------------------
    # IMAGE PREVIEW
    # --------------------------------------------------
    def preview_image(self, item):
        if not item:
            self.preview.clear()
            return

        person_dir = os.path.join(REFERENCE_DIR, item.text())
        images = [
            f for f in os.listdir(person_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]

        if not images:
            self.preview.setText("No images")
            return

        img_path = os.path.join(person_dir, images[0])
        pix = QPixmap(img_path)
        self.preview.setPixmap(
            pix.scaled(
                self.preview.width(),
                self.preview.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
        )
