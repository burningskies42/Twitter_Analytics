from nltk.corpus import twitter_samples
from tweets_to_df import tweet_json_to_df
from sentiment_mod import sentiment
import pandas as pd

# print(twitter_samples.fileids())
# tokenized = twitter_samples.tokenized('negative_tweets.json')

import easygui

dataset_path = easygui.fileopenbox(default='*.JSON',filetypes=[["*.pickle", "Binary files"]])
if dataset_path == None:
    quit()
else:
    df = tweet_json_to_df(dataset_path)

catg = {}
for each in df['text']:
    # print(each,sentiment(each))
    catg[each] = sentiment(each)


df = pd.DataFrame.from_dict(catg,orient='index')
print(df.head())
df.to_csv('sent_ml.csv',sep=';')