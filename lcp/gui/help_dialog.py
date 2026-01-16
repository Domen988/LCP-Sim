
import os
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QPushButton, QMessageBox, QLabel
)
from PyQt6.QtCore import Qt

class HelpDialog(QDialog):
    def __init__(self, readme_path: str, parent=None):
        super().__init__(parent)
        self.readme_path = readme_path
        self.setWindowTitle("LCP-Sim Workflow Guide")
        self.resize(600, 500)
        
        self.layout = QVBoxLayout(self)
        
        # Editor
        self.editor = QTextEdit()
        self.editor.setReadOnly(True)
        self.layout.addWidget(self.editor)
        
        # Buttons
        self.btn_layout = QHBoxLayout()
        
        self.btn_edit = QPushButton("Edit")
        self.btn_edit.clicked.connect(self.toggle_edit)
        
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        
        self.btn_layout.addStretch()
        self.btn_layout.addWidget(self.btn_edit)
        self.btn_layout.addWidget(self.btn_close)
        
        self.layout.addLayout(self.btn_layout)
        
        self.is_editing = False
        self.load_content()
        
    def load_content(self):
        if os.path.exists(self.readme_path):
            try:
                with open(self.readme_path, "r") as f:
                    self.content = f.read()
                self.editor.setPlainText(self.content)
            except Exception as e:
                self.editor.setPlainText(f"Error loading README: {e}")
        else:
            self.content = "README.md not found."
            self.editor.setPlainText(self.content)
            
    def toggle_edit(self):
        if not self.is_editing:
            # Enter Edit Mode
            self.is_editing = True
            self.editor.setReadOnly(False)
            self.editor.setStyleSheet("background-color: #333333; color: white;") # Visual cue? Theme dependent.
            # Assuming dark theme, standard edit color is fine.
            
            self.btn_edit.setText("Save")
            self.btn_close.setText("Cancel")
            self.btn_edit.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold;")
            
            # Disconnect close
            self.btn_close.clicked.disconnect()
            self.btn_close.clicked.connect(self.cancel_edit)
            
        else:
            # Save (Exit Edit Mode)
            self.save_content()
            
    def save_content(self):
        new_content = self.editor.toPlainText()
        try:
            with open(self.readme_path, "w") as f:
                f.write(new_content)
            
            self.content = new_content
            self.is_editing = False
            self.editor.setReadOnly(True)
            self.editor.setStyleSheet("")
            
            self.btn_edit.setText("Edit")
            self.btn_close.setText("Close")
            self.btn_edit.setStyleSheet("")
            
            self.btn_close.clicked.disconnect()
            self.btn_close.clicked.connect(self.close)
            
            QMessageBox.information(self, "Saved", "Readme updated successfully.")
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))
            
    def cancel_edit(self):
        # Revert
        self.editor.setPlainText(self.content)
        self.is_editing = False
        self.editor.setReadOnly(True)
        self.editor.setStyleSheet("")
        
        self.btn_edit.setText("Edit")
        self.btn_close.setText("Close")
        self.btn_edit.setStyleSheet("")
        
        self.btn_close.clicked.disconnect()
        self.btn_close.clicked.connect(self.close)
