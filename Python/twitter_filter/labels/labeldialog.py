from sys import argv
from PyQt4 import QtGui, QtCore


class Window(QtGui.QDialog):
    def __init__(self,progress):
        super(QtGui.QDialog, self).__init__()
        self.setGeometry(1400, 400, 0, 0)
        self.setFixedSize(280, 120)
        self.setWindowTitle("Labeler")
        self.setWindowIcon(QtGui.QIcon('pylogo.png'))
        self.setWindowFlags(QtCore.Qt.Sheet)
        self.label = None
        # self.statusTip()
        self.set_dialog(progress)



    def set_dialog(self,progress):
        button_size = [70, 30]

        # Label geometry
        myfont = QtGui.QFont()
        myfont.setPixelSize(20)
        lbl = QtGui.QLabel('News or not ?', self)
        lbl.setFont(myfont)
        lbl.resize(lbl.minimumSizeHint())
        lbl.move(70, 5)
        self.progress_val = progress

        self.progress = QtGui.QProgressBar(self)
        self.progress.setGeometry(10, 85, 270, 15)
        self.progress.setValue(self.progress_val)

        # News button geometry
        btnnews = QtGui.QPushButton("News", self)
        btnnews.clicked.connect(self.return_one)
        btnnews.resize(button_size[0],button_size[1])
        btnnews.move(10, 40)
        btnnews.setStatusTip('Mark tweet as "News"')

        # Not-News button geometry
        btnnot = QtGui.QPushButton("Not News", self)
        btnnot.clicked.connect(self.return_two)
        btnnot.resize(button_size[0],button_size[1])
        btnnot.move(100, 40)
        btnnot.setStatusTip('Mark tweet as "Not News"')

        # Not-News button geometry
        btnnot = QtGui.QPushButton("Not Found", self)
        btnnot.clicked.connect(self.return_three)
        btnnot.resize(button_size[0],button_size[1])
        btnnot.move(190, 40)
        btnnot.setStatusTip('Tweet was not found')

    def return_one(self):
        self.done(1)
        self.close()

    def return_two(self):
        self.done(2)
        self.close()

    def return_three(self):
        self.done(3)
        self.close()


def get_labeler(progress):
    app = QtGui.QApplication(argv)
    GUI = Window(progress)
    GUI.setWindowState(GUI.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
    GUI.activateWindow()
    return GUI.exec()
