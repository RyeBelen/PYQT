import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Labels");
        self.setGeometry(700, 300, 500,500)
        
        self.button = QPushButton("Click Me", self)
        self.label  = QLabel("Hello", self)

        self.initUI()

    def initUI(self):
        #button is only a local variable so we need to prefix it with self
        self.button.setGeometry(150,200,200,100)
        self.button.setStyleSheet("font-size: 30px;")
        self.button.clicked.connect(self.on_click)
    
        # in order to connect the button to a function, we use the clicked signal
        # a signal is ommitted when there is an interaction with the button

        # with buttons, you need a signal that is connected to a slot
        # signal = event, slot = action to be done

        self.label.setGeometry(150,300,200,100)
        self.label.setStyleSheet("font-size: 50px;")

    def on_click(self):
        print("Clicked ampota")
        self.button.setText("Clicked")
        self.button.setDisabled(True) 

        #change the label when clicked
        self.label.setText("Byebye!")


def main():
    app = QApplication(sys.argv) 
    window = MainWindow()
    window.show() #shows only once
    sys.exit(app.exec_()) #built in execute method, waits for it


if __name__ == "__main__":
    main()