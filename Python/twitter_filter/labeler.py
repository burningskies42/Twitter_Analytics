from tweet_tk.tweets_to_df import tweet_json_to_df
import pandas as pd
from selenium import webdriver
from easygui import buttonbox,fileopenbox
from os import path
import random

# Create List in csv
file = fileopenbox()
file_name=file.split('\\')[-1].split('_')[0] +'_labeled_tweets.csv'

if not path.isfile(file_name):
    print('new file')
    df = pd.DataFrame(tweet_json_to_df(file))
    id_df = pd.DataFrame(df.index.values,columns=['id'])
    id_df['label'] = 0
    id_df.set_index('id',inplace=True)
    id_df.to_csv(file_name)
else:
    print('existing file')

label_dict = {}
df = pd.DataFrame.from_csv(file_name)

ids = [id for id in df.index.values if df['label'][id] == 0]
random.shuffle(ids)

for id in df.index.values:
    print(id,df['label'][id],df['label'][id] == 0)



if len(ids) == 0:
    print('all labeled')
    quit()

# Open chrome browser
chromedriver = "C:/Program Files (x86)/Google/Chrome/Application/chromedriver.exe"
driver = webdriver.Chrome(chromedriver)
driver.maximize_window()

for i in ids[:5]:
    tweet_address = 'https://twitter.com/anyuser/status/' + str(i)
    driver.get(tweet_address)

    choice = buttonbox('News or not',choices=['News','Not News'])
    if choice == None:
        driver.close()
        quit()

    print(choice)
    label_dict[i] = (1 if choice == 'News' else 2)

driver.close()

for key,val in label_dict.items():
    df['label'][key] = val
    df.to_csv(file_name)
# for key,val in label_dict.items():
#     print(key,val)
