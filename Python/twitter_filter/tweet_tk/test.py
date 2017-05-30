import tweepy
import pandas as pd

api_key = pd.read_pickle('auth/twitter_auth_key.pickle')

auth = tweepy.OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
auth.set_access_token(api_key['access_token'], api_key['access_secret'])

api = tweepy.API(auth)

tweets = tweepy.Cursor(api.search,q="Amazon",count=100,result_type="popular",include_entities=True,lang="en").items()
new_ids = set([tweet.id for tweet in tweets])
# print(len(new_ids))
# for id in new_ids:
#     print(id,type(id))

old_ids = set(pd.Series.from_csv('hist_tweets.csv',sep=';').tolist())

all_ids = old_ids.union(new_ids)

print(len(all_ids))

all_ids = pd.Series(list(all_ids))
all_ids.to_csv('hist_tweets.csv',sep=';')
