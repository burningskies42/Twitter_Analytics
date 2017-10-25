from sys import argv
from PyQt4 import QtGui, QtCore


class Window(QtGui.QDialog):
   def __init__(self,ignore_terms=[]):
      super(QtGui.QDialog, self).__init__()
      self.setGeometry(600, 300, 0, 0)
      self.setFixedSize(700, 400)
      self.setWindowTitle("Labeler")
      self.setWindowIcon(QtGui.QIcon('pylogo.png'))
      self.setWindowFlags(QtCore.Qt.Sheet)
      self.setWindowTitle('Twitter Seatch API 0.2')
      self.label = None
      self.statusTip()
      self.ignore_terms = ignore_terms

      self.set_dialog(ignore_terms)

   def set_dialog(self,ignore_terms):
      button_size = [70, 30]
      self.ignore_terms = ignore_terms

      # Label geometry
      title_font = QtGui.QFont()
      title_font.setPixelSize(20)

      label_font = QtGui.QFont()
      label_font.setPixelSize(18)

      # Searchbox geometry
      search_lbl = QtGui.QLabel('search', self)
      search_lbl.setFont(label_font)
      search_lbl.resize(search_lbl.minimumSizeHint())
      search_lbl.move(25, 43)

      search_box = QtGui.QLineEdit(self)
      search_box.move(85, 45)
      search_box.resize(200, 20)

      console_box = QtGui.QPlainTextEdit(self)
      console_box.hide()

      # ignorelist geometry
      ignore_lbl = QtGui.QLabel('ignore', self)
      ignore_lbl.setFont(label_font)
      ignore_lbl.resize(search_lbl.minimumSizeHint())
      ignore_lbl.move(340, 43)

      ignore_text = QtGui.QLineEdit(self)
      ignore_text.move(400, 45)
      ignore_text.resize(200, 20)

      def add_ignore_term():
         text = str(ignore_text.text())
         if len(ignore_text.text()) > 0:
            if text not in self.ignore_terms:
               ignore_list.addItem(ignore_text.text())
               ignore_text.setText('')
            else:
               pass

      def get_ignore_terms():
         items = []
         for i in range(ignore_list.count()):
            items.append(ignore_list.item(i))
         self.ignore_terms = items

      ignore_btn = QtGui.QPushButton("Add", self)
      ignore_btn.clicked.connect(add_ignore_term)
      ignore_btn.resize(200, 30)
      ignore_btn.move(400, 70)
      ignore_btn.setStatusTip('Search Twitter Databases')

      ignore_list = QtGui.QListWidget(self)
      ignore_list.move(400, 105)
      ignore_list.resize(200, 100)
      ignore_list.clicked.connect(get_ignore_terms)
      for each in self.ignore_terms:
         ignore_list.addItem(each)

      def start_search(self):
         ignore_list.close()
         console_box.show()
         console_box.move(25,105)
         console_box.resize(650,270)
         ignore_btn.setDisabled(True)
         ignore_text.setDisabled(True)
         search_box.setDisabled(True)
         btnSearch.setDisabled(True)


      # Search button geometry
      btnSearch = QtGui.QPushButton("Search", self)
      btnSearch.clicked.connect(start_search)
      btnSearch.resize(200, 30)
      btnSearch.move(85, 70)
      btnSearch.setStatusTip('Search Twitter Databases')

   def return_two(self):
      self.done(2)
      self.close()



def show_filter_gui(ignore_items=[]):
   app = QtGui.QApplication(argv)
   GUI = Window(ignore_items)
   GUI.setWindowState(GUI.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
   GUI.activateWindow()
   return GUI.exec()


