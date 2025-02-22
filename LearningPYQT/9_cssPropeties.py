import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QHBoxLayout

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Labels");
    
        # with main window widget there's already a specified format
        # so we will add a layout manager to a central widget and this will be added to the main window

        self.button1 = QPushButton("#1")
        self.button2 = QPushButton("#2")
        self.button3 = QPushButton("#3")
        self.initUI()

    def initUI(self):

        #central widet
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        #layout manager
        hbox = QHBoxLayout()
        hbox.addWidget(self.button1)
        hbox.addWidget(self.button2)   
        hbox.addWidget(self.button3)

        # add layout manager to central widget
        central_widget.setLayout(hbox)

        self.button1.setObjectName("button1")
        self.button2.setObjectName("button2")
        self.button3.setObjectName("button3")

        # CSS SHIT
        self.setStyleSheet("""
            QPushButton{
                font-size: 40px;
                font-family: Arial;
                padding: 15px 75px;
                margin: 15px;
                border: 2px solid;
                border-radius: 15px;
            }

            QPushButton#button1{
                background-color: hsl(0, 100%, 64%);
            }
            QPushButton#button2{
                background-color: hsl(122, 100%, 64%);
            }
            QPushButton#button3{
                background-color: hsl(204, 100%, 64%);
            }

            QPushButton#button1:hover{
                background-color: hsl(0, 100%, 84%);
            }
            QPushButton#button2:hover{
                background-color: hsl(122, 100%, 84%);
            }
            QPushButton#button3:hover{
                background-color: hsl(204, 100%, 84%);
            }
            
            """)
        
        

def main():
    app = QApplication(sys.argv) 
    window = MainWindow()
    window.show() #shows only once
    sys.exit(app.exec_()) #built in execute method, waits for it


if __name__ == "__main__":
    main()