from pyqt

def ShowGroupROIFunction(self):
    dialog = SGROIWidget_ui.Ui_ShowGroupWidget()
    if dialog.exec_():
        print(dialog.roiGroups)
The other one:

...

class Ui_ShowGroupWidget(QtGui.QDialog):
    def __init__(self):
        QtGui.QDialog.__init__(self)
        self.setupUi(self)
        self.roiGroups = {}
        self.Submit.clicked.connect(self.submitclose)

    def setupUi(self, ShowGroupWidget):
        #sets up Submit button

    def submitclose(self):
        #do whatever you need with self.roiGroups
        self.accept()