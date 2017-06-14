from pandas import DataFrame, read_pickle, Series
from tweepy import OAuthHandler, API
from os import getcwd
from difflib import SequenceMatcher
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import easygui

from tweet_tk.tweets_to_df import tweet_json_to_df
# Calculate how similiar two string are in the [0:1] range
def similar(a, b):
   return SequenceMatcher(None, a, b).ratio()


# connect to twitter server
def tweet_text(id_list):
   api_key = read_pickle(getcwd() + '\\tweet_tk\\auth\\twitter_auth_key.pickle')
   auth = OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
   auth.set_access_token(api_key['access_token'], api_key['access_secret'])
   api = API(auth)
   print('\nQuerying tweet ids from server:')

   i = 0
   tweets_df = DataFrame()
   step = 100

   stop_words = set(stopwords.words('english'))

   while i < len(id_list):

      # query to get list of STATUS objects from ids
      ids_chunk = id_list[i:min(i + step, len(id_list))]

      print('queried', i, 'to', min(i + step, len(id_list)))
      tweets_chunk = API.statuses_lookup(api, list(ids_chunk))

      # removing stop words and URLs from texts
      tweets_dict = {}
      for tweet in tweets_chunk:
         tweets_dict[tweet._json['id']] = ' '.join([w for w in word_tokenize(str(tweet._json['text']))
                                                    if w not in stop_words
                                                    and w.find('http')== -1])

      tweets_dict = DataFrame.from_dict(tweets_dict, orient='index')
      tweets_dict.columns = ['text']
      tweets_df = tweets_df.append(tweets_dict)

      i += step

   return tweets_df


def remove_similiar_tweets(df,sim_upper_lim = 0.7):
   ids = df.index.values
   # df = df[:10]

   df_w_text = tweet_text(ids)
   print(df_w_text.head())
   quit()

   df_w_text['text'] = df_w_text['text'].apply(lambda x: x.replace('\n', ''))

   # df = df.join(df_w_text)

   new_df = DataFrame(columns=['id', 'text'])
   new_df.set_index('id', inplace=True)

   last_len = 0
   scanned_len = 0

   for indexI, rowI in df_w_text.iterrows():
      scanned_len +=1
      if last_len < len(new_df):
         last_len = len(new_df)
         print(scanned_len,'/',last_len)

      addrow = True

      for indexJ, rowJ in new_df.iterrows():
         similiarity = similar(str(rowI['text']), str(rowJ['text']))

         # When as similiar tweet already in new df, chuck if its older
         # if older - replace, otherwise ignore
         if similiarity > sim_upper_lim:

            addrow = False
            break

      if addrow:
         s = Series({'text': rowI['text']})
         s.name = indexI
         new_df = new_df.append(s)

   # new_df.to_csv('labeled_filtered.csv', sep=';')

   if 'text' in df.columns:
      df.drop('text',axis=1,inplace = True)
   new_df = new_df.join(df,how='inner')
   new_df['text'] = new_df['text'].apply(lambda x : x.replace('\n','').replace('\r',''))
   new_df = new_df[['text']]

   return(new_df)


df_path = easygui.fileopenbox(default='*.json',filetypes=['*.json','*.csv'])

file_type = df_path.split('.')[len(df_path.split('.'))-1]

if file_type == 'csv':
   df = DataFrame.from_csv(df_path, sep=';')
   df = df[:100]
   df = remove_similiar_tweets(df)
   df_path = df_path.replace('.csv', '_filtered.csv')
   df.to_csv(df_path, sep=';')

elif file_type == 'json':
   df = tweet_json_to_df(df_path)
   df = remove_similiar_tweets(df)
   df_path = df_path.replace('.json','_filtered.csv')
   df.to_csv(df_path,sep=';')


# df = DataFrame.from_csv(getcwd() + '\labels\Amazon_labeled_tweets.csv.new_collection', sep=';')





