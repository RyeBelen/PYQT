import sys
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, 
                             QWidget, QVBoxLayout, QHBoxLayout, QGridLayout) # deals with layout managers

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Labels");
        self.setGeometry(700, 300, 500,500)
        self.initUI()

    #anything that deals with userint we do here
    def initUI(self):
        #create widget -> add layout manager to widget -> add widget to main window
        central_widget = QWidget(self) #constructor for QWidget
        self.setCentralWidget(central_widget) #sets the central widget


        #LABELS
        label1 = QLabel("Label 1", self)
        label2 = QLabel("Label 2", self)
        label3 = QLabel("Label 3", self)
        label4 = QLabel("Label 4", self)
        label5 = QLabel("Label 5", self)

        label1.setStyleSheet("background-color: red")
        label2.setStyleSheet("background-color: green")
        label3.setStyleSheet("background-color: blue")
        label4.setStyleSheet("background-color: yellow")
        label5.setStyleSheet("background-color: purple")

        label1.setAlignment(Qt.AlignCenter)
        label2.setAlignment(Qt.AlignCenter)
        label3.setAlignment(Qt.AlignCenter)
        label4.setAlignment(Qt.AlignCenter)
        label5.setAlignment(Qt.AlignCenter)


        '''
        #LAYOUT MANAGERS
        # VERTICAL LAYOUT
        vbox = QVBoxLayout()
        vbox.addWidget(label1)
        vbox.addWidget(label2)  
        vbox.addWidget(label3)
        vbox.addWidget(label4)
        vbox.addWidget(label5)

        central_widget.setLayout(vbox)

        # HORIZONTAL LAYOUT

        hbox = QHBoxLayout()

        hbox.addWidget(label1)
        hbox.addWidget(label2)
        hbox.addWidget(label3)
        hbox.addWidget(label4)
        hbox.addWidget(label5)

        central_widget.setLayout(hbox)

        '''

        #GRID LAYOUT
        gbox = QGridLayout()
                            #ROW, COLUMN
        gbox.addWidget(label1, 0, 0) 
        gbox.addWidget(label2, 0, 1)
        gbox.addWidget(label3, 0, 2)
        gbox.addWidget(label4, 1, 0)
        gbox.addWidget(label5, 1, 1)

        central_widget.setLayout(gbox)


def main():
    app = QApplication(sys.argv) 
    window = MainWindow()
    window.show() 
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()