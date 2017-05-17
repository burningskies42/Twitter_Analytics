import sys
from PyQt4 import QtGui, QtCore

class Window(QtGui.QDialog):
    def __init__(self):
        super(QtGui.QDialog, self).__init__()
        self.setGeometry(800, 500, 0, 0)
        self.setFixedSize(265,120)
        self.setWindowTitle("Labeler")
        self.setWindowIcon(QtGui.QIcon('pylogo.png'))
        self.setWindowFlags(QtCore.Qt.Sheet)
        self.label = None
        self.set_dialog()

    def set_dialog(self):

        # Label geometry
        myfont = QtGui.QFont()
        myfont.setPixelSize(20)
        lbl = QtGui.QLabel('News or not ?',self)
        lbl.setFont(myfont)
        lbl.resize(lbl.minimumSizeHint())
        lbl.move(50,15)

        # News button geometry
        btnnews = QtGui.QPushButton("News", self)
        btnnews.clicked.connect(self.return_one)
        btnnews.resize(btnnews.minimumSizeHint())
        btnnews.move(25, 60)
        btnnews.setStatusTip('Mark tweet as "News"')

        # Not-News button geometry
        btnnot = QtGui.QPushButton("Not News", self)
        btnnot.clicked.connect(self.return_two)
        btnnot.resize(btnnot.minimumSizeHint())
        btnnot.move(140, 60)
        btnnot.setStatusTip('Mark tweet as "Not News"')

    def return_one(self):
        self.done(1)
        self.close()

    def return_two(self):
        self.done(2)
        self.close()

def open_labeler():
    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    GUI.setWindowState(GUI.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
    GUI.activateWindow()
    return GUI.exec()
