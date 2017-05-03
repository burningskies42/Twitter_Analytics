import tweepy
import pandas as pd
import json

api_key = pd.read_pickle('twitter_auth_key.pickle')
# consumer_key = '***'
# consumer_secret = '***'
# access_token = '***'
# access_token_secret = '***'

auth = tweepy.OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
auth.set_access_token(api_key['access_token'], api_key['access_secret'])

api = tweepy.API(auth)

results = api.search(q="amazon",lang='en',rpp=100)
results_df = pd.DataFrame(results)

df = pd.DataFrame()

for st in results:
    json_str = json.dumps(st._json)
    line = pd.read_json(json_str,typ='series')
    df = df.append(line,ignore_index=True)

print(df[['retweet_count','id_str','text']])
