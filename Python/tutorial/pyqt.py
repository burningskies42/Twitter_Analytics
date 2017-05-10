import sys
from PyQt4 import QtGui, QtCore
from tweet_tk.tweets_to_df import tweet_json_to_df
import pandas as pd
from selenium import webdriver
from easygui import buttonbox,fileopenbox

class Window(QtGui.QDialog):
    def __init__(self):
        super(QtGui.QDialog, self).__init__()
        self.setGeometry(800, 500, 0, 0)
        self.setFixedSize(265,120)
        self.setWindowTitle("Labeler")
        self.setWindowIcon(QtGui.QIcon('pylogo.png'))
        self.setWindowFlags(QtCore.Qt.Sheet)
        self.label = None
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
        # self.close_application()

    def set_label_two(self):
        self.label = 2
        # self.close_application()

    # def close_application(self):
    #     sys.exit()


def run():
    # Create List in csv
    file = fileopenbox()
    df = pd.DataFrame(tweet_json_to_df(file))
    id_df = pd.DataFrame(df.index.values,columns=['id'])
    id_df['label'] = 0
    id_df.set_index('id',inplace=True)
    id_df.to_csv('labeled_tweets.csv')


    label_dict = {}
    df = pd.DataFrame.from_csv('labeled_tweets.csv')

    # Open chrome browser
    chromedriver = "C:/Program Files (x86)/Google/Chrome/Application/chromedriver.exe"
    driver = webdriver.Chrome(chromedriver)
    driver.maximize_window()

    app = QtGui.QApplication(sys.argv)


    for i in df.index.values[:10]:
        tweet_address = 'https://twitter.com/anyuser/status/' + str(i)
        driver.get(tweet_address)
        # GUI = Window()
        # GUI.setWindowState(GUI.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
        # GUI.activateWindow()
        # choice = buttonbox('News or not',choices=['News','Not News'])
        # sys(app.exec_())
        choice = buttonbox('News or not', choices=['News', 'Not News'],root_height=10,root_width=20)
        print(choice)
        # label_dict[i] = (1 if choice == 'News' else 2)

    driver.close()

    for key, val in label_dict.items():
        print(key, val)

run()