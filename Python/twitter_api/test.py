import pandas as pd
from tweepy import OAuthHandler, API
import tweets_to_df

# extract as list of tweet ids

# dataset = pd.read_pickle('amazon_dataset.pickle') #from pickle

dataset = tweets_to_df.tweet_json_to_df('amazon_dataset.json') #from json
# dataset = dataset[:100]
ids = list(dataset['id_str'])

# connect to twitter server
api_key = pd.read_pickle('twitter_auth_key.pickle')
auth = OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
auth.set_access_token(api_key['access_token'],api_key['access_secret'])
api = API(auth)

tweets_df = pd.DataFrame()
i=0
while i <= len(ids):
    # query to get list of STATUS objects from ids

    ids_chunk = ids[i:min(i + 100, len(ids))]

    print('queried',i,'to',min(i + 100, len(ids)))
    tweets_chunk = API.statuses_lookup(api, ids_chunk)
    i+=100

    for tweet in tweets_chunk:
        se = pd.Series(tweet._json)
        tweets_df = tweets_df.append(se,ignore_index=True)

tweets_df.set_index('id',inplace=True)
tweets_df[['id','created_at','retweet_count']].to_csv('retweet_count.csv',sep=';')
print('saved to retweet_count.csv')

