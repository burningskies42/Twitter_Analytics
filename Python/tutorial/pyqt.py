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
        self.setWindowModality(QtCore.Qt.WindowModal)
        # extractAction = QtGui.QAction("&Exit", self)
        # extractAction.setShortcut("Ctrl+Q")
        # extractAction.setStatusTip('Leave The App')
        # extractAction.triggered.connect(self.close_application)

        # self.statusBar()
        #
        # mainMenu = self.menuBar()
        # fileMenu = mainMenu.addMenu('&File')
        # fileMenu.addAction(extractAction)

        self.home()

    def home(self):
        myfont = QtGui.QFont()
        myfont.setPixelSize(20)

        lbl = QtGui.QLabel('News or not ?',self)
        lbl.setFont(myfont)
        lbl.resize(lbl.minimumSizeHint())
        lbl.move(50,15)


        btnnews = QtGui.QPushButton("News", self)
        btnnews.clicked.connect(self.set_label_one)
        btnnews.resize(btnnews.minimumSizeHint())
        btnnews.move(25, 60)
        btnnews.setStatusTip('Mark tweet as "News"')

        btnnot = QtGui.QPushButton("Not News", self)
        btnnot.clicked.connect(self.set_label_two)
        btnnot.resize(btnnot.minimumSizeHint())
        btnnot.move(140, 60)
        btnnot.setStatusTip('Mark tweet as "Not News"')

        self.show()

    def set_label_one(self):
        self.label = 1
        self.close()

    def set_label_two(self):
        self.label = 2
        self.close()

    def get_label(self):
        return self.label
        self.close()

    def close_application(self):
        sys.exit()


def run():



    app = QtGui.QApplication(sys.argv)
    GUI = Window()
    print(GUI.result())

    sys.exit(app.exec_())


run()