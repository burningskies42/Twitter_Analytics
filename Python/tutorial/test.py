import pandas as pd
import numpy as np
import tweepy

'''
dataset = pd.read_pickle('amazon_db.pickle')
api_key = pd.read_pickle('twitter_auth_key.pickle')

auth = tweepy.OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
auth.set_access_token(api_key['access_token'],api_key['access_secret'])
api = tweepy.API(auth)

# print(dataset.columns.values)

ids_w_RT = dataset['id_str']
i=0
lists = []

while i < len(ids_w_RT):
    ls = ids_w_RT.iloc[i:i+100]
    lists.append(ls)
    i +=100
    # print('added ',i,'to',i+100)


i=0
for ls in lists:
    i += 1
    df = api.statuses_lookup(list(lists[1]),)
    df = pd.DataFrame(df)
    df.to_pickle('df_statuses_'+str(i)+'.pickle')
    print('pickled part',i)

'''

for i in range(1,42):
    df = pd.read_pickle('df_statuses_'+str(i)+'.pickle')
    df = df.append(df)

df.to_csv('try_csv.csv',sep=';')
