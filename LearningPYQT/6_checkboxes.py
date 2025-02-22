import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QCheckBox
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Labels");
        self.setGeometry(700, 300, 500,500)
        
        self.checkbox = QCheckBox("Mahilig ka ba mag ano?", self)


        self.initUI()

    def initUI(self):
        self.checkbox.setGeometry(10,0, 500, 100)
        self.checkbox.setStyleSheet("font-size: 30px;"
                                    "font-family: Arial;")
        #self.checkbox.setChecked(True) keeps the checkbox checked

        self.checkbox.stateChanged.connect(self.checkbox_stateChange)

    def checkbox_stateChange(self, state):
        print(state)

        if state == Qt.Checked:
            print("Mahilig ka mag ano")
        else:
            print("Di ka mahilig mag ano")

def main():
    app = QApplication(sys.argv) 
    window = MainWindow()
    window.show() #shows only once
    sys.exit(app.exec_()) #built in execute method, waits for it


if __name__ == "__main__":
    main()