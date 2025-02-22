import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt



class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Labels");
        self.setGeometry(700, 300, 500,500)

        label = QLabel("Hello", self)
        label.setFont(QFont("Arial", 40))
        label.setGeometry(0,0,500,100)
        label.setStyleSheet("color: blue;"
                            "background-color: yellow;"
                            "border-style: solid;"
                            "font-style: italic;"
                            "font-weight: bold;"
                            "text-decoration: underline;")
        
        # VeRTICAL ALIGNMENT
        #label.setAlignment(Qt.AlignTop)
        #label.setAlignment(Qt.AlignBottom)
        #label.setAlignment(Qt.AlignVCenter)

        # HORIZONTAL ALIGNMENT
        #label.setAlignment(Qt.AlignLeft)
        #label.setAlignment(Qt.AlignRight)
        #label.setAlignment(Qt.AlignHCenter)

        # Cewnter Allignment
        #label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        #label.setAlignment(Qt.AlignHCenter | Qt.AlignBottom)
        #label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        label.setAlignment(Qt.AlignCenter)

def main():
    app = QApplication(sys.argv) 
    window = MainWindow()
    window.show() #shows only once
    sys.exit(app.exec_()) #built in execute method, waits for it


if __name__ == "__main__":
    main()