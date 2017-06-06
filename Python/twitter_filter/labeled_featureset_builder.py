import pandas as pd
from tweepy import OAuthHandler,API
from feature_tk.features import tweets_to_featureset
from nltk.corpus import stopwords

def fetch_tweets_by_ids(id_list):
   api_key = pd.read_pickle('tweet_tk\\auth\\twitter_auth_key.pickle')
   auth = OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
   auth.set_access_token(api_key['access_token'], api_key['access_secret'])
   api = API(auth)
   df = None
   i = 0

   # file = open('amazon_labeled.json', 'w')
   while i < len(id_list):
      u_limit = min(i+100,len(id_list))
      ids_chunk = id_list[i:u_limit]
      tweets_chunk = API.statuses_lookup(api, list(ids_chunk))
      for tweet in tweets_chunk:
         new_row = pd.Series(tweet._json)

         if df is None:
            df = pd.DataFrame(index=id_list,columns=new_row.keys())
         try:
            df.loc[tweet._json['id']] = new_row
         except Exception as e:
            print(new_row)
            quit()

         # json.dump(tweet._json,file)
         # file.write('\n\n')
         # print(new_row['id'])
      print('Retrieved tweets:',i,'to',u_limit)
      i+=100
   # file.close()
   print('------------------------------------')
   # remove empty rows (deleted tweets)
   df.dropna(inplace=True,how='all')

   return df

def fetch_tweet(tweet_id):
   api_key = pd.read_pickle('tweet_tk\\auth\\twitter_auth_key.pickle')
   auth = OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
   auth.set_access_token(api_key['access_token'], api_key['access_secret'])
   api = API(auth)

   status = api.get_status(tweet_id)
   ser = pd.Series(status._json)
   tweet = ser.to_frame()

   return tweet

def open_and_join(file,save_to_file = False, with_sentiment = True,with_timing=True):
   label_df = pd.DataFrame.from_csv(file, sep=';')
   ids = label_df.index.values
   df = fetch_tweets_by_ids(ids)

   featureset = tweets_to_featureset(df,with_sentiment,with_timing)
   labeled_featureset = pd.concat([featureset,label_df],axis=1,join='outer')
   labeled_featureset.dropna(axis = 0,how = 'any',inplace = True)

   stop_words = set(stopwords.words('english'))

   all_words_news = {}
   all_words_not = {}
   for i,row in labeled_featureset.iterrows():
      if row['label'] == 1:
         for word in [w for w in row['words'] if w not in stop_words and len(w)>1]:
            if word in all_words_news.keys():
               all_words_news[word] += 1
            else:
               all_words_news[word] = 1
      elif row['label'] == 2:
         for word in [w for w in row['words'] if w not in stop_words and len(w)>1]:
            if word in all_words_not.keys():
               all_words_not[word] += 1
            else:
               all_words_not[word] = 1

   # quit()
   #
   print('------------------------------------')
   print('news words:')
   for key,val in sorted(all_words_news.items(),key=lambda x:x[1],reverse=True)[:50]:
      print(key,val)

   print('\nnot-news words:')
   for key, val in sorted(all_words_not.items(), key=lambda x: x[1], reverse=True)[:50]:
      print(key,val)

   print('------------------------------------')
   # labeled_featureset.drop(['words', 'words_no_url'], axis=1, inplace=True)

   if save_to_file:
      labeled_featureset = labeled_featureset[labeled_featureset['label']!= 3]
      labeled_featureset.to_csv('labeled_featureset.csv',sep=';')

   return labeled_featureset

