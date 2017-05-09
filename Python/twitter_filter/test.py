# from tweet_tk.tweets_to_df import tweet_json_to_df
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from easygui import buttonbox

# # Create List in csv
# file = fileopenbox()
# df = pd.DataFrame(tweet_json_to_df(file))
#
# id_df = pd.DataFrame(df.index.values,columns=['id'])
# id_df['label'] = 0
# id_df.set_index('id',inplace=True)
# id_df.to_csv('labeled_tweets.csv')

import tkinter.messagebox

top = tkinter.Tk()
def label_box():
    B1 = tkinter.Button(top, text = "foo", command = label_box)
    B2 = tkinter.Button(top, text = "bar", command = label_box)

    B1.grid(row=1,column=1)
    B2.grid(row=1,column=2)

    top.mainloop()

label_box()
quit()

df = pd.DataFrame.from_csv('labeled_tweets.csv')

driver = webdriver.Chrome('C:/Program Files (x86)/Google/Chrome/Application/chromedriver.exe')
driver.maximize_window()
label_dict = {}

for i in df.index.values[:10]:
    tweet_address = 'https://twitter.com/anyuser/status/' + str(i)
    driver.get(tweet_address)
    # choice = buttonbox('News or not',choices=['News','Not News'])
    choice = Mbox('Labeler', 'Is it news ?', 4)
    print(choice)
    label_dict[i] = (1 if choice == 'News' else 2)

driver.close()

for key,val in label_dict.items():
    print(key,val)