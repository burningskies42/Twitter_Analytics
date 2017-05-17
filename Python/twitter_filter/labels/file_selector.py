from PyQt4 import QtGui

def file_open():
    name = QtGui.QFileDialog.getOpenFileName('Open File')
    file = open(name, 'r')

file_open()