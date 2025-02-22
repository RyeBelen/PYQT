import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLineEdit # a fucking textbox,

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Labels");
        self.setGeometry(700, 300, 500,500)
        self.button = QPushButton("Submit", self)
        self.line_edit = QLineEdit(self)


        self.initUI()

    def initUI(self):
        self.line_edit.setGeometry(10,10, 200, 40)
        self.line_edit.setStyleSheet("font-size: 25px;"
                                    "font-family: Arial;")
        self.line_edit.setPlaceholderText("Enter your name")
        
        self.button.setGeometry(10,60, 200, 40)
        self.button.setStyleSheet("font-size: 20px;"
                                    "font-family: Arial;")
        
        self.button.clicked.connect(self.submit)
    
    def submit(self):
        text = self.line_edit.text()
        print(f"Hallur, {text}!")

def main():
    app = QApplication(sys.argv) 
    window = MainWindow()
    window.show() #shows only once
    sys.exit(app.exec_()) #built in execute method, waits for it


if __name__ == "__main__":
    main()