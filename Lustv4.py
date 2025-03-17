import sys
import os
import time
import datetime
import threading
import pyttsx3
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QLineEdit, QPushButton, QButtonGroup, QRadioButton, QLabel, QMessageBox,
    QGroupBox, QSplitter, QComboBox, QFileDialog, QMenu, QAction
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QColor, QTextCursor, QTextCharFormat, QIcon
from llama_cpp import Llama

class LlamaThread(QThread):
    response_signal = Signal(str, bool)  # str: token, bool: is_code flag
    finished_signal = Signal()
    error_signal = Signal(str)

class AIInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Interface")
        self.setGeometry(100, 100, 800, 600)

        self.llm = Llama()
        self.engine = pyttsx3.init()

        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # Option buttons
        option_layout = QHBoxLayout()
        self.voice_button = QPushButton("Change Voice")
        self.voice_button.setMenu(self.create_voice_menu())
        option_layout.addWidget(self.voice_button)

        self.llm_button = QPushButton("Change LLM")
        self.llm_button.clicked.connect(self.change_llm)
        option_layout.addWidget(self.llm_button)

        main_layout.addLayout(option_layout)

        # Setup button
        self.setup_button = QPushButton("Setup")
        self.setup_button.setStyleSheet("background-color: rgba(255, 255, 255, 0.5);")
        self.setup_button.setMenu(self.create_setup_menu())
        main_layout.addWidget(self.setup_button, alignment=Qt.AlignRight)

        # Text edit and input
        self.text_edit = QTextEdit()
        main_layout.addWidget(self.text_edit)

        self.input_line = QLineEdit()
        main_layout.addWidget(self.input_line)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_input)
        main_layout.addWidget(self.send_button)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def create_voice_menu(self):
        menu = QMenu()
        voices = self.engine.getProperty('voices')
        for voice in voices:
            action = QAction(voice.name, self)
            action.triggered.connect(lambda checked, v=voice: self.change_voice(v))
            menu.addAction(action)
        return menu

    def change_voice(self, voice):
        self.engine.setProperty('voice', voice.id)

    def change_llm(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select LLM File", "", "Model Files (*.bin *.pt)")
        if file_path:
            self.llm.load_model(file_path)

    def create_setup_menu(self):
        menu = QMenu()

        # Voice options
        voice_menu = QMenu("Change Voice", self)
        voices = self.engine.getProperty('voices')
        for voice in voices:
            action = QAction(voice.name, self)
            action.triggered.connect(lambda checked, v=voice: self.change_voice(v))
            voice_menu.addAction(action)
        menu.addMenu(voice_menu)

        # LLM model change
        llm_action = QAction("Change LLM Model", self)
        llm_action.triggered.connect(self.change_llm)
        menu.addAction(llm_action)

        # Internet search option
        internet_action = QAction("Enable Internet Search", self)
        internet_action.setCheckable(True)
        internet_action.triggered.connect(self.toggle_internet_search)
        menu.addAction(internet_action)

        # Safety protocols
        safety_action = QAction("Enable Safety Protocols", self)
        safety_action.setCheckable(True)
        safety_action.setChecked(True)
        menu.addAction(safety_action)

        return menu

    def toggle_internet_search(self, checked):
        if checked:
            QMessageBox.information(self, "Internet Search", "Internet search enabled.")
        else:
            QMessageBox.information(self, "Internet Search", "Internet search disabled.")

    def send_input(self):
        input_text = self.input_line.text()
        self.text_edit.append(f"User: {input_text}")
        self.input_line.clear()

        # Simulate LLM response
        response = self.llm.generate_response(input_text)
        self.text_edit.append(f"LLM: {response}")

def main():
    app = QApplication(sys.argv)
    window = AIInterface()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
