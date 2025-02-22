import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QRadioButton, QButtonGroup

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Labels");
        self.setGeometry(700, 300, 500,500)

        self.radiobutton1 = QRadioButton("Visa", self)
        self.radiobutton2 = QRadioButton("Paypal", self) 
        self.radiobutton3 = QRadioButton("Mastercard", self)
        self.radiobutton4 = QRadioButton("In-store", self)
        self.radiobutton5 = QRadioButton("Online", self)



        self.buttonGroup1 = QButtonGroup(self)
        self.buttonGroup2 = QButtonGroup(self)

        self.initUI()

    def initUI(self):

        # all radio buttons unless explicityly stated are part of the same group
        self.radiobutton1.setGeometry(10,0, 500, 100)
        self.radiobutton2.setGeometry(10,50, 500, 100)
        self.radiobutton3.setGeometry(10,100, 500, 100)

        self.radiobutton4.setGeometry(10,150, 500, 100)
        self.radiobutton5.setGeometry(10,200, 500, 100)

        # apply a stylesheet to a group of widgets
        self.setStyleSheet("QRadioButton{"
                           "font-size: 40px;"
                           "font-family: Arial;"
                           "padding: 10px;"
                           "}")
        
        self.buttonGroup1.addButton(self.radiobutton1)
        self.buttonGroup1.addButton(self.radiobutton2)
        self.buttonGroup1.addButton(self.radiobutton3)

        self.buttonGroup2.addButton(self.radiobutton4)
        self.buttonGroup2.addButton(self.radiobutton5)

        self.radiobutton1.toggled.connect(self.radio_button_changed)
        self.radiobutton2.toggled.connect(self.radio_button_changed)
        self.radiobutton3.toggled.connect(self.radio_button_changed)
        self.radiobutton4.toggled.connect(self.radio_button_changed)
        self.radiobutton5.toggled.connect(self.radio_button_changed)
    
    def radio_button_changed(self):

        radio_Button = self.sender()
        if radio_Button.isChecked():
            print(f"{radio_Button.text()} is selected")


def main():
    app = QApplication(sys.argv) 
    window = MainWindow()
    window.show() #shows only once
    sys.exit(app.exec_()) #built in execute method, waits for it


if __name__ == "__main__":
    main()