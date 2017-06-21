from tweepy import OAuthHandler, API
import pandas as pd

def fetch_tweets_by_ids(id_list):
   api_key = pd.read_pickle('tweet_tk\\auth\\twitter_auth_key.pickle')
   auth = OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
   auth.set_access_token(api_key['access_token'], api_key['access_secret'])
   api = API(auth)
   df = None
   i = 0

   # file = open('amazon_labeled.json', 'w')

   # Iterrate through all ids, 100 at a time
   while i < len(id_list):
      u_limit = min(i+100,len(id_list))
      ids_chunk = id_list[i:u_limit]
      tweets_chunk = API.statuses_lookup(api, list(ids_chunk))
      for tweet in tweets_chunk:
         new_row = pd.Series(tweet._json)
         # print(new_row)
         if df is None:
            df = pd.DataFrame(columns=new_row.keys())
         # try:
         # print(df)
         # df.loc[tweet._json['id']] = new_row
         df = df.append([new_row])
         # except Exception as e:
         #    print(new_row)
         #    quit()

         # json.dump(tweet._json,file)
         # file.write('\n\n')
         # print(new_row['id'])
      print('Retrieved tweets:',i,'to',u_limit)
      i+=100
   # file.close()
   print('------------------------------------')
   # remove empty rows (deleted tweets)
   df.dropna(inplace=True,how='all')
   df.set_index('id_str',inplace=True)

   return df