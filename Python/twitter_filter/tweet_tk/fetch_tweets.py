from tweepy import OAuthHandler, API
import pandas as pd
'''
Receives a list of tweet ids. Each tweet id comes in the form Int64. The ids must be handled
properly, by being converted to strings. Strings dont lose precision.
The twitter API allows to fetch up to 100 tweets per query, so the list must be split to
chunks of 100 ids each.
@:param(id_list) - list of tweets ids
'''
def fetch_tweets_by_ids(id_list):
   api_key = pd.read_pickle('tweet_tk\\auth\\twitter_auth_key.pickle')
   auth = OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
   auth.set_access_token(api_key['access_token'], api_key['access_secret'])
   api = API(auth)
   df = None
   i = 0

   # Iterrate through all ids, 100 at a time
   while i < len(id_list):
      u_limit = min(i+100,len(id_list))
      ids_chunk = id_list[i:u_limit]
      tweets_chunk = API.statuses_lookup(api, list(ids_chunk))

      # Convert each tweet to Series and append to DF
      for tweet in tweets_chunk:
         new_row = pd.Series(tweet._json)

         if df is None:
            df = pd.DataFrame(columns=new_row.keys())

         df = df.append([new_row])

      print('Retrieved tweets:',i,'to',u_limit)
      i+=100
   print('------------------------------------')

   # remove empty rows (deleted tweets)
   df.dropna(inplace=True,how='all')
   df.set_index('id_str',inplace=True)

   return df