import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QLabel
from PyQt5.QtGui import QPixmap # PROVIDES FUNCITONALITY FOR LOADING, DISPLAYING AND MANIPULATING IMAGES

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Images");
        self.setGeometry(700, 300, 500,500)

        label = QLabel(self)
        label.setGeometry(0,0, 250, 250)

        pixmap = QPixmap("LearningPYQT/img/burg.png") # loads using pixmap
        label.setPixmap(pixmap) # sets the image to the label
        # since image doesnt scale to the label, we need to scale it
        label.setScaledContents(True)

        '''

        #top left align
        label.setGeometry(0,                             # X
                          0,                             # Y
                          label.width(),                 # WIDTH
                          label.height())                # HEIGHT
        #right align
        label.setGeometry(self.width() - label.width(),  # X
                          0,                             # Y
                          label.width(),                 # WIDTH
                          label.height())                # HEIGHT

        #bottom left align
        label.setGeometry(0,                             # X
                          self.height() - label.height(),# Y
                          label.width(),                 # WIDTH
                          label.height())                # HEIGHT
        #bottom right align
        label.setGeometry(self.width() - label.width(),  # X
                          self.height() - label.height(),# Y
                          label.width(),                 # WIDTH
                          label.height())                # HEIGHT
        '''

        #center align
        label.setGeometry(self.width() // 2 - label.width() // 2,  # X
                          self.height() // 2 - label.height() // 2, # Y
                          label.width(),                 # WIDTH
                          label.height())                # HEIGHT


def main():
    app = QApplication(sys.argv) 
    window = MainWindow()
    window.show() #shows only once
    sys.exit(app.exec_()) #built in execute method, waits for it

if __name__ == "__main__":
    main()