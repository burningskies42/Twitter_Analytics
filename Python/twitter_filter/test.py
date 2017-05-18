import pandas as pd
from tweepy import OAuthHandler,API
import json


label_df = pd.DataFrame.from_csv('labels/Amazon_labeled_tweets.csv',sep = ';')
ids = label_df.index.values

def fetch_tweets_by_ids(id_list):
   api_key = pd.read_pickle('tweet_tk\\auth\\twitter_auth_key.pickle')
   auth = OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
   auth.set_access_token(api_key['access_token'], api_key['access_secret'])
   api = API(auth)

   i = 0
   df = pd.DataFrame()
   file = open('amazon_labeled.json', 'w')
   while i < len(ids):
      u_limit = min(i+100,len(ids))
      ids_chunk = ids[i:u_limit]
      tweets_chunk = API.statuses_lookup(api, list(ids_chunk))
      for tweet in tweets_chunk:
         new_row = pd.Series(tweet._json)
         new_df = df.append(new_row,ignore_index=True)
         json.dump(tweet._json,file)
         file.write('\n\n')
         print(new_row['id'])

      print('added',i,'to',u_limit)
      i+=100
   file.close()
   return df

def open_and_join(file):
   feature_df = pd.DataFrame.from_csv(file,sep=';',index_col=1)
   print(feature_df.head())

open_and_join('feature_sets/labeled.csv')

