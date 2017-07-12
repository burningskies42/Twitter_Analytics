from re import sub, MULTILINE
from difflib import SequenceMatcher
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime, timezone
from textblob import Blobber
from textblob_aptagger import PerceptronTagger
import pandas as pd
import easygui
from os import system, getcwd
from tweepy import OAuthHandler, API
from csv import reader
from codecs import iterdecode
import pickle
from string import punctuation

from tweet_tk.retweet_fetcher import retweet_cnt
from tweet_tk.bots import add_suspects, is_suspect
from tweet_tk.emoticons_parser import emoticons_score
from tweet_tk.tweet_sentiment import sentiment
from tweet_tk.tweets_to_df import tweet_json_to_df

from time import time,strftime,gmtime

tb = Blobber(pos_tagger=PerceptronTagger())

import bs4 as bs
import urllib.request
from pandas import DataFrame, Series


# Get ~100 most popular urls from wikipedia
def most_pop_urls_wiki():
   try:
      source = urllib.request.urlopen('https://en.wikipedia.org/wiki/List_of_most_popular_websites').read()
      soup = bs.BeautifulSoup(source, 'lxml')

      table = soup.find(class_="wikitable sortable")
      table_rows = table.find_all('tr')

      rating_table = DataFrame()

      for tr in table_rows:
         td = tr.find_all('td')
         row = [i.text for i in td]
         if len(row) > 0:
            rating_table = rating_table.append(Series(row), ignore_index=True)

      rating_table.columns = ['Site', 'Domain', 'Alexa', 'SimilarWeb', 'Type', 'Country']

      with open('feature_tk/wiki_top_100.pkl','wb') as fid:
         pickle.dump(rating_table,fid)
         fid.close()



   except Exception as e:
      with open('feature_tk/wiki_top_100.pkl', 'rb') as fid:
         rating_table = pickle.load(fid)
         fid.close()

      print('Error when downloading from wikipedia, used local copy')

   finally:
      # print(rating_table['Domain'])
      return set(rating_table['Domain'].tolist())


def most_pop_urls_moz():

   try:
      source = urllib.request.urlopen('https://moz.com/top500').read()
      soup = bs.BeautifulSoup(source, 'lxml')

      table = soup.find(class_="table table-bordered table-zebra")
      table_rows = table.find_all('tr')
      rating_table = DataFrame()

      for tr in table_rows:
         td = tr.find_all('td')
         row = [i.text.replace('\n','').replace(' ','') for i in td]
         if len(row) > 0:
            rating_table = rating_table.append(Series(row), ignore_index=True)

      rating_table.columns = ['Rank','Root Domain','Linking Root Domains',
                              'External Links','Domain mozRank','Domain mozTrust','Change']
      rating_table.set_index('Rank',inplace=True)

      # Save local copy
      with open('feature_tk/moz_top_500.pkl', 'wb') as fid:
         pickle.dump(rating_table,fid)
         fid.close()

   except Exception as e:
      print('Error when downloading from MOZ, using local copy')
      fid = open('feature_tk/moz_top_500.pkl','rb')
      rating_table = pickle.load(fid)
      fid.close()

   finally:

      lst = set(rating_table['Root Domain'].tolist())
      lst = [url.replace('/', '') for url in lst]
      return lst


# generate list of most popular websites
most_pop_urls_wiki = list(most_pop_urls_wiki())
most_pop_urls_moz = most_pop_urls_moz()
print('downloaded most popular domains\n')


# CURRENTLY NOT IN USE !!!
def similar(df):
   sent_df = df.to_frame()
   sent_df['similiarity'] = ''

   for i, sentence in sent_df.iterrows():
      max_ratio = 0
      for j, another in sent_df.iterrows():

         # if different tweets, compare to max
         if i != j:
            simil = SequenceMatcher(None, sentence, another).ratio()
            if simil == 1:
               sent_df['similiarity'][i] = 1
               # sent_df['similiarity'][j] = 1
               break

            elif simil > max_ratio:
               max_ratio = simil

      sent_df['similiarity'][i] = max_ratio
   return sent_df['similiarity']


def tokenize_and_filter(sentence):
   sentence = word_tokenize(sentence)
   stop_words = set(stopwords.words('english'))
   words = []
   for w in sentence:
      if w.lower() not in stop_words:
         words.append(w)
   # words = [w for w in sentence if w.lower() not in stop_words]
   tagged = tb.pos_tagger(words)
   # tagged = pos_tag(words)
   return tagged


def has_pronoun(text):
   tagged_words = tb(text)
   dc = [x[1] for x in tagged_words.pos_tags]
   return ('PRP' in dc)


def count_upper(text):
   return round(sum(1 for c in text if c.isupper()) / len(text), 3)


def clear_urls(text):
   clear_text = sub(r'https?:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text, flags=MULTILINE)
   clear_text = clear_text.replace('\n', ' ')
   return clear_text


def account_age(user_created_at):
   creataion_date = datetime.strptime(user_created_at, '%a %b %d %H:%M:%S %z %Y')
   return (datetime.now(timezone.utc) - creataion_date).days


def is_wiki_url(entity):
   if len(entity) > 0:
      urls = [x['expanded_url'] for x in entity['urls']]

      wiki_flag = False
      url = None
      for each in urls:
         url = each.split('/')[2]
         if len(url.split('www.')) > 1:
            url = url.split('www.')[1]

         if url in most_pop_urls_wiki:
            wiki_flag = True

   return wiki_flag


def is_moz_url(entity):
   if len(entity) > 0:
      urls = [x['expanded_url'] for x in entity['urls']]

      moz_flag = False
      url = None
      for each in urls:
         url = each.split('/')[2]
         if len(url.split('www.')) > 1:
            url = url.split('www.')[1]

         if url in most_pop_urls_moz:
            moz_flag = True

   return moz_flag

def count_most_pop(words,most_pop_words):
   cnt = 0
   for word in words:
      if word in most_pop_words:
         cnt+=1

   return cnt

def clear_punctuation(string):
   string = string.replace('\r', '').replace('\n', '')
   string = string.translate(punctuation)
   return string

def flatten_users(df):
   users_df = df['user'].apply(Series)
   users_df.columns = 'user_' + users_df.columns
   df = pd.concat([df.drop(['user'], axis=1), users_df], axis=1)
   return df


# Message content related features
def msg_feature_df(df, with_sentiment=True, with_timing=True,with_most_pop_words=True):
   df_msg = pd.DataFrame(index=df.index)
   # df_msg = []
   # df_msg['id'] = df.index.values
   # df_msg.set_index('id', inplace=True)
   all_words = {}
   stop_words = set(stopwords.words('english'))

   start = time()
   df_msg['words'] = df['text'].apply(lambda x : clear_punctuation(x))
   df_msg['words'] = df_msg['words'].apply(lambda x: word_tokenize(x))
   df_msg['num_stop_words'] = df_msg['words'].apply(lambda x: len(x))
   dur = time() - start
   if with_timing: print('tokenize words:', dur)

   if with_most_pop_words:
      df_msg['label'] = df['label']

      for index,row in df_msg.iterrows() :
            words = [w.lower() for w in row['words'] if len(w)>1 and w.lower() not in stop_words and str.find(w,'/')==-1]
            for word in words:
               if word in all_words.keys():
                  all_words[word] += 1
               else:
                  all_words[word] = 1

      df_msg.drop('label',axis=1,inplace=True)

      # get the 5000 most common words in all the tweets
      all_words = sorted(all_words.items(), key=lambda x: x[1], reverse=True)
      all_words = [w[0] for w in all_words]

      news_words_w = {}
      news_len = len(df[df['label']==1])
      not_news_len = len(df[df['label'] == 2])

      print('~~~~~~~~~~~~~~~~~~~~~~~~')
      print('News samples:',news_len)
      print('Not news samples:', not_news_len)
      print('~~~~~~~~~~~~~~~~~~~~~~~~')

      # save to file
      with open('news_words.pkl','wb') as fid:
         pickle.dump(all_words,fid)
         fid.close()



   start = time()
   df_msg['words'] = df_msg['words'].apply(lambda x: [w.lower() for w in x
                                                      if w.lower() not in stop_words
                                                      and len(w)>1
                                                      and w.lower().find('http') == -1
                                                      and w.lower().find('//') == -1])
   dur = time() - start
   if with_timing: print('filter out stop words:', dur)

   start = time()
   df_msg['num_stop_words'] = df_msg['num_stop_words'] - df_msg['words'].apply(lambda x: len(x))
   dur = time() - start
   if with_timing: print('counting stop words:', dur)

   start = time()
   df_msg['words_no_url'] = df['text'].apply(lambda x: clear_urls(x))
   df_msg['words_no_url'] = df_msg['words_no_url'].apply(lambda x : x.replace('\r',' ').replace('\n',' '))
   dur = time() - start
   if with_timing: print('clear urls:', dur)

   start = time()
   df_msg['duplicate'] = df_msg['words_no_url'].duplicated(keep=False)
   dur = time() - start
   if with_timing: print('duplicates:', dur)

   start = time()
   bot_suspects = set(df['user_id'][df_msg['duplicate'] == True])
   add_suspects(bot_suspects, with_text=with_timing)
   dur = time() - start
   if with_timing: print('add_suspects:', dur)

   start = time()
   df_msg['len_characters'] = df['text'].apply(lambda x: len(x))
   dur = time() - start
   if with_timing: print('len_characters:', dur)

   start = time()
   df_msg['num_words'] = df_msg['words'].apply(lambda x: len(x))
   dur = time() - start
   if with_timing: print('num_words:', dur)

   start = time()
   df_msg['has_question_mark'] = df['text'].apply(lambda x: x.find('?') != -1)
   dur = time() - start
   if with_timing: print('has_question_mark:', dur)

   start = time()
   df_msg['has_exclamation_mark'] = df['text'].apply(lambda x: x.find('!') != -1)
   dur = time() - start
   if with_timing: print('has_exclamation_mark:', dur)

   start = time()
   df_msg['has_multi_quest_exclam'] = df['text'].apply(lambda x: (x.count('?') > 1 or x.count('!') > 1))
   dur = time() - start
   if with_timing: print('has_multi_quest_exclam:', dur)

   start = time()
   df_msg['emotji_sent_score'] = df['text'].apply(lambda x: emoticons_score(x))
   dur = time() - start
   if with_timing: print('emotji_sent_score:', dur)

   start = time()
   df_msg['has_pronoun'] = df['text'].apply(lambda x: has_pronoun(x))
   dur = time() - start
   if with_timing: print('has_pronoun:', dur)

   start = time()
   df_msg['count_upper'] = df['text'].apply(lambda x: count_upper(x))
   dur = time() - start
   if with_timing: print('count_upper:', dur)

   start = time()
   df_msg['has_hashtag'] = df['text'].apply(lambda x: x.find('#') != -1)
   dur = time() - start
   if with_timing: print('has_hashtag:', dur)

   start = time()
   df_msg['retweet_count'] = df['retweet_count']
   dur = time() - start
   if with_timing: print('retweet_count:', dur)

   start = time()
   df_msg['urls_wiki'] = df['entities'].apply(lambda x: is_wiki_url(x))
   df_msg['urls_moz'] = df['entities'].apply(lambda x: is_moz_url(x))
   dur = time() - start
   if with_timing: print('urls:', dur)

   if with_sentiment:
      start = time()
      sentm = df['text'].apply(lambda x: sentiment(x))
      sentm = sentm.apply(pd.Series)
      sentm.columns = ['class', 'conf']
      df_msg['senitment'] = sentm['class']
      df_msg['senitment_conf'] = sentm['conf']
      dur = time() - start
      if with_timing: print('senitment:', dur)

   return df_msg, all_words


# User related features
def usr_feature_df(df, with_timing=True):
   df_user = pd.DataFrame(index=df.index)
   # df_user['id'] = df.index
   # df_user.set_index('id',inplace=True)

   start = time()
   df_user['reg_age'] = df['user_created_at'].apply(lambda x: account_age(x))
   dur = time() - start
   if with_timing: print('reg_age:', dur)

   start = time()
   df_user['status_cnt'] = df['user_statuses_count']
   dur = time() - start
   if with_timing: print('statuses_count:', dur)

   start = time()
   df_user['followers_cnt'] = df['user_followers_count']
   dur = time() - start
   if with_timing: print('followers_count:', dur)

   start = time()
   df_user['friends_cnt'] = df['user_friends_count']
   dur = time() - start
   if with_timing: print('friends_count:', dur)

   start = time()
   df_user['verified'] = df['user_verified']
   dur = time() - start
   if with_timing: print('verified:', dur)

   start = time()
   df_user['has_desc'] = df['user_description'].apply(lambda x: x != None)
   dur = time() - start
   if with_timing: print('has_desc:', dur)

   start = time()
   df_user['has_url'] = df['user_url'].apply(lambda x: x != None)
   dur = time() - start
   if with_timing: print('has_url:', dur)

   start = time()
   df_user['msg_p_day'] = df['user_statuses_count'] / df_user['reg_age']
   df_user['msg_p_day'] = df_user['msg_p_day'].apply(lambda x : round(x,3))
   dur = time() - start
   if with_timing: print('msg_p_day:', dur)

   return df_user


# Builds a featureset df from a captured_tweets_df
def tweets_to_featureset(df, with_sentiment=True, with_timing=True,with_most_pop_words=True):

   # convert json object USER to columns
   df = flatten_users(df)

   # build feature table for different feature categories
   msg_feat_df,all_words = msg_feature_df(df, with_sentiment, with_timing,with_most_pop_words)
   # msg_feat_df.drop(['words', 'words_no_url'], axis=1, inplace=True)
   usr_feat_df = usr_feature_df(df, with_timing)

   if with_timing:
      print('\nValue Frequencies:')
      print(msg_feat_df['retweet_count'].value_counts()[:10])
      # retweets_freq = msg_feat_df['retweet_count'].value_counts(,)
      # print(retweets_freq)

   df = pd.concat([msg_feat_df, usr_feat_df], axis=1)

   # start_time = time()
   # with open('C:/Users/Leon/Documents/Masterarbeit/Python/twitter_filter/classifiers/words_as_features/Words.pickle','rb') as fid:
   #    loaded_words = pickle.load(fid)
   #    fid.close()
   #
   # print('loaded_words',len(loaded_words))
   #
   # df_words = pd.DataFrame(columns=loaded_words, index=df.index)
   #
   # cnt = 0
   # for i, row in df.iterrows():
   #    for w in df_words.columns.values:
   #       df_words.loc[i][w] = (w in row['words'])
   #
   #    cnt+=1
   #    if (cnt % 10) == 0:
   #       ecum_time = (time()-start_time)
   #       perc_completed = round(cnt*100/len(df),2)
   #       speed = perc_completed/ecum_time
   #       #
   #       rem_time = int((100-perc_completed)/speed)
   #       rem_time = strftime('%H:%M:%S', gmtime(rem_time))
   #       print(str(perc_completed)+'%','time remainig',rem_time)
   #
   # df = pd.concat([df, df_words], axis=1)
   # df.to_csv('foobar.csv',sep = ';')
   #
   # print('building word features',round(time()-start_time,2))

   # Obsolete, since retweets are counted from captured df
   # df = pd.concat([df, retweets], axis=1)
   # df.set_index('id_str')
   return df      #,all_words


def single_tweet_features(tweet_id):
   api_key = pd.read_pickle('tweet_tk\\auth\\twitter_auth_key.pickle')
   auth = OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
   auth.set_access_token(api_key['access_token'], api_key['access_secret'])
   api = API(auth)

   status = api.get_status(tweet_id)
   ser = pd.Series(status._json)
   print(ser)
   quit()

   features = {}
   features['duplicate'] = is_suspect(tweet_id)

   return features


# GUI and file selector for "tweets_to_featureset"
def features_from_file():
   # Clear console and print greeting
   system('CLS')
   print('''
                    ***********************************************************************
                    *                                                                     *
                    *              Feature Set Builder for Twitter Datasets               *
                    *                              Version 0.1                            *
                    *             Author: Leon Edelmann        Copyright 2017 (c)         *
                    *                                                                     *
                    ***********************************************************************
    ''')

   print('loaded toolkit packages ...')

   # open target tweets dataset
   default_path = getcwd() + '\captured_tweets\*.JSON'
   dataset_path = easygui.fileopenbox(default=default_path, filetypes=[["*.pickle", "Binary files"]])
   if dataset_path == None:
      quit()

   # Extract dataset name from path
   file_name = dataset_path.split('\\')[len(dataset_path.split('\\')) - 1].split('_dataset.json')[0]

   # start = time()
   df = tweet_json_to_df(dataset_path)

   # build feature table for different feature categories
   print('\nValue Frequencies:')

   feature_df = tweets_to_featureset(df)

   feature_df.to_csv('feature_tk\\feature_sets\\' + file_name + '_feature_set.csv', sep=';', index=True)
   print('\nSaved to ...\\' + file_name + '_feature_set.csv')
