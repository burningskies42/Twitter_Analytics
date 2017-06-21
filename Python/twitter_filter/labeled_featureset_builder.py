import pandas as pd
from tweepy import OAuthHandler,API
from feature_tk.features import tweets_to_featureset
from nltk.corpus import stopwords
import pickle
from tweet_tk.fetch_tweets import fetch_tweets_by_ids

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
   print(file)
   label_df = pd.DataFrame.from_csv(file, sep=';')
   label_df.index = label_df.index.map(str)
   label_df = label_df[label_df['label']!= 0]
   print(len(label_df),'labels found in file')

   ids = label_df.index.values
   df = fetch_tweets_by_ids(ids)

   featureset = tweets_to_featureset(df,with_sentiment,with_timing)
   # print(type(featureset.index.values[0]))
   # print(type(label_df.index.values[0]))

   labeled_featureset = pd.concat([featureset,label_df],axis=1,join='inner')
   # labeled_featureset.to_csv('fuck.fuck',sep = ";")
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

      with open('labeled_featureset.pkl','wb') as  fid:
         pickle.dump(labeled_featureset,fid)
         fid.close()

      # pd.DataFrame(labeled_featureset).to_pickle()

   return labeled_featureset

