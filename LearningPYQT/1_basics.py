import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel
from PyQt5.QtGui import QIcon
'''
    BOILET PLATE CODE FOR A BASIC WINDOW

    import sys
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()

    def main():
        app = QApplication(sys.argv) 
        window = MainWindow()
        window.show() #shows only once
        sys.exit(app.exec_()) #built in execute method, waits for it


    if __name__ == "__main__":
        main()
'''

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Title"); # title
        self.setGeometry(700, 300, 500,500) # x, y, width, height
        self.setWindowIcon(QIcon("LearningPYQT/img/burg.png")) #icon

def main():
    app = QApplication(sys.argv) 
    window = MainWindow()
    window.show() #shows only once
    sys.exit(app.exec_()) #built in execute method, waits for it


if __name__ == "__main__":
    main()