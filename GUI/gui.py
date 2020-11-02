import sys
import urllib.request
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import uic

main_ui = uic.loadUiType("GUI/untitled.ui")[0]

class RegisterWindow(QDialog):
    def __init__(self, parent):
        super(RegisterWindow, self).__init__(parent)
        register_ui = ''


class MainWindow(QMainWindow, main_ui) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)

        # START 버튼 클릭 시 이벤트 발생
        self.start_button.clicked.connect(self.startButtonFunction)

    def startButtonFunction(self):


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    app.exec_() 