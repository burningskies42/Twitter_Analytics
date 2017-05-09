import time
from PyQt4 import QtCore, QtGui

class Window(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        layout = QtGui.QVBoxLayout(self)
        self.label = QtGui.QLabel(self)
        layout.addWidget(self.label)
        self.buttonStart = QtGui.QPushButton('Start', self)
        self.buttonStart.clicked.connect(self.handleStart)
        layout.addWidget(self.buttonStart)
        self.buttonStop = QtGui.QPushButton('Stop', self)
        self.buttonStop.clicked.connect(self.handleStop)
        layout.addWidget(self.buttonStop)
        self._running = False

    def handleStart(self):
        self.buttonStart.setDisabled(True)
        self._running = True
        while self._running:
            self.label.setText(str(time.clock()))
            QtGui.qApp.processEvents()
            time.sleep(0.05)
        self.buttonStart.setDisabled(False)

    def handleStop(self):
        self._running = False

if __name__ == '__main__':

    import sys
    app = QtGui.QApplication(sys.argv)
    window = Window()
    window.setGeometry(500, 300, 200, 100)
    window.show()
    sys.exit(app.exec_())