from tweet_tk.tweets_to_df import tweet_json_to_df
import pandas as pd
from easygui import fileopenbox
from os import path
import random
from labels.labeldialog import get_labeler
from labels.chromedriver import get_chrome
from PyQt4 import QtGui

# Create List in csv
file = fileopenbox()
file_name = file.split('\\')[-1].split('_')[0] +'_labeled_tweets.csv'

if not path.isfile(file_name) :
    print('new file')
    df = pd.DataFrame(tweet_json_to_df(file))
    id_df = pd.DataFrame(df.index.values,columns=['id'])
    id_df['label'] = 0
    id_df.set_index('id',inplace=True)
    id_df.to_csv(file_name,sep=';')

else:
    print('existing file')

label_dict = {}
df = pd.DataFrame.from_csv(file_name,sep=';')
print(len(df))
# ids = [id for id in df.index.values if df['label'][id] == 0]
ids = df[df['label']==0].index.values
random.shuffle(ids)

print('Labeled',str(len(df) - len(ids))+'/'+str(len(df)))

if len(ids) == 0:
    print('all labeled or empty list')
    quit()

# Open chrome browser
driver = get_chrome()
driver.maximize_window()


cnt = 1
curr_list = ids[:100]
for i in curr_list:
    tweet_address = 'https://twitter.com/anyuser/status/' + str(i)
    driver.get(tweet_address)

    # opens labeling promt
    percent_completed = int(100*cnt/len(curr_list))
    choice = get_labeler(percent_completed)
    if choice == 0:
        driver.close()
        quit()

    print(choice,str(cnt)+'/'+str(len(curr_list)))
    label_dict[i] = choice
    cnt += 1

driver.close()

for key,val in label_dict.items():
    df['label'][key] = val
    df.to_csv(file_name,sep=';')
