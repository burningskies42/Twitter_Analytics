import tweepy
import pandas as pd
import json

api_key = pd.read_pickle('auth/twitter_auth_key.pickle')

auth = tweepy.OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
auth.set_access_token(api_key['access_token'], api_key['access_secret'])

api = tweepy.API(auth)

i=0
df = pd.DataFrame()
since_id = 0

while i < 10:

    results = api.search(q="Amazon",lang='en',count=100,result_type='popular',since_id=since_id)
    results_df = pd.DataFrame(results)

    for st in results:
        json_str = json.dumps(st._json)
        line = pd.read_json(json_str,typ='series')
        df = df.append(line,ignore_index=True)

    i+=1
    since_id = max(df['id_str'].tolist())
    print(i,since_id)

print(df[['retweet_count','id_str','text']])
df.drop('text',axis=1,inplace=True)
df.to_csv('hist_tweets.csv',sep=';')