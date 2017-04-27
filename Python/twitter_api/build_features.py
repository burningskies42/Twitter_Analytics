# my modules
from tweets_to_df import tweet_json_to_df
from emoticons_parser import emoticons_score
from retweet_fetcher import retweet_cnt
from url_parser import most_pop_urls

# generate list of most popular websites
try:
    most_pop_urls = list(most_pop_urls()['Domain'])
    print('downloaded most popular domains\n')
except Exception as e:
    print(e)
    quit()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import time

from datetime import datetime,timezone
from textblob import Blobber
from textblob_aptagger import PerceptronTagger
tb = Blobber(pos_tagger=PerceptronTagger())
import easygui


dataset_path = easygui.fileopenbox(default='*.JSON',filetypes=[["*.pickle", "Binary files"]])
if dataset_path == None:
    quit()

# Extract dataset name from path
file_name = dataset_path.split('\\')[len(dataset_path.split('\\'))-1].split('_dataset.json')[0]
# print(file_name)

# the class will produce and contain all features
# not in use
class tweet_fetures():
    def __init__(self,tweet,tweet_index):
        # msg. features
        self.id = tweet_index
        self.text = tweet['text']
        self.words = self.text.split()
        self.len_characters = len(self.text)
        self.len_words = len(self.words)
        self.has_question_mark = self.text.find('?') != -1
        self.has_exclamation_mark = self.text.find('!') != -1
        self.has_multi_quest_exclam = self.text.count('?') > 1 or self.text.count('!') > 1
        self.emotji_sent_score = emoticons_score(self.text)

    def features_as_series(self):
        row = pd.Series({
            'text' : self.text,
            'words' : self.words,
            'len_characters' : self.len_characters,
            'len_words' : self.len_words,
            'has_question_mark' : self.has_question_mark,
            'has_exclamation_mark' : self.has_exclamation_mark,
            'has_multi_quest_exclam' : self.has_multi_quest_exclam,
            'emotji_sent_score' : self.emotji_sent_score,
        }, index = [self.id])
        return row

def tokenize_and_filter(sentence):
    sentence = word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    words = []
    for w in sentence:
        if w.lower() not in stop_words:
                words.append(w)
    # words = [w for w in sentence if w.lower() not in stop_words]
    tagged = pos_tag(words)
    return tagged

def has_pronoun(text):
    tagged_words = tb(text)
    dc = [x[1] for x in tagged_words.pos_tags]
    return ('PRP' in dc)

def count_upper(text):
    return sum(1 for c in text if c.isupper())/len(text)

def get_urls(entity):
    if len(entity) > 0:
        urls = [x['expanded_url'] for x in entity['urls']]

        flag = False
        url = None
        for each in urls:
            url = each.split('/')[2]
            if len(url.split('www.')) > 1:
                url = url.split('www.')[1]

            if url in most_pop_urls:
                flag = True

    return flag

def msg_feature_df(df):
    df_msg = pd.DataFrame()
    df_msg['id'] = df.index
    df_msg.set_index('id', inplace=True)

    start = time.time()
    df_msg['words'] = df['text'].apply(lambda x : word_tokenize(x))
    dur = time.time() - start
    print('tokenize words:',dur)

    start = time.time()
    stop_words = set(stopwords.words('english'))
    df_msg['words'] = df_msg['words'].apply(lambda x : [w for w in x if w.lower() not in stop_words])
    dur = time.time() - start
    print('filter out stop words:',dur)

    start = time.time()
    df_msg['len_characters'] = df['text'].apply(lambda x : len(x))
    dur = time.time() - start
    print('len_characters:', dur)

    start = time.time()
    df_msg['num_words'] = df_msg['words'].apply(lambda x : len(x))
    dur = time.time() - start
    print('num_words:', dur)

    start = time.time()
    df_msg['has_question_mark'] = df['text'].apply(lambda x : x.find('?') != -1)
    dur = time.time() - start
    print('has_question_mark:', dur)

    start = time.time()
    df_msg['has_exclamation_mark'] = df['text'].apply(lambda x : x.find('!') != -1)
    dur = time.time() - start
    print('has_exclamation_mark:', dur)

    start = time.time()
    df_msg['has_multi_quest_exclam'] = df['text'].apply(lambda x : (x.count('?') > 1 or x.count('!') > 1))
    dur = time.time() - start
    print('has_multi_quest_exclam:', dur)

    start = time.time()
    df_msg['emotji_sent_score'] = df['text'].apply(lambda x : emoticons_score(x))
    dur = time.time() - start
    print('emotji_sent_score:', dur)

    start = time.time()
    df_msg['has_pronoun'] = df['text'].apply(lambda x : has_pronoun(x))
    dur = time.time() - start
    print('has_pronoun:', dur)

    start = time.time()
    df_msg['count_upper'] = df['text'].apply(lambda x : count_upper(x))
    dur = time.time() - start
    print('count_upper:', dur)

    start = time.time()
    df_msg['has_hashtag'] = df['text'].apply(lambda x: x.find('#') != -1)
    dur = time.time() - start
    print('has_hashtag:', dur)

    start = time.time()
    df_msg['urls'] = df['entities'].apply(lambda x: get_urls(x))
    dur = time.time() - start
    print('urls:', dur)

    return df_msg

def usr_feature_df(df):
    df_user = pd.DataFrame()
    df_user['id'] = df.index
    df_user.set_index('id',inplace=True)

    def account_age(user_created_at):
        creataion_date = datetime.strptime(user_created_at, '%a %b %d %H:%M:%S %z %Y')
        return (datetime.now(timezone.utc) - creataion_date).days

    start = time.time()
    df_user['reg_age'] = df['user_created_at'].apply(lambda x: account_age(x))
    dur = time.time() - start
    print('reg_age:', dur)

    start = time.time()
    df_user['status_cnt'] = df['user_statuses_count']
    dur = time.time() - start
    print('statuses_count:', dur)

    start = time.time()
    df_user['followers_cnt'] = df['user_followers_count']
    dur = time.time() - start
    print('followers_count:', dur)

    start = time.time()
    df_user['friends_cnt'] = df['user_friends_count']
    dur = time.time() - start
    print('friends_count:', dur)

    start = time.time()
    df_user['verified'] = df['user_verified']
    dur = time.time() - start
    print('verified:', dur)

    start = time.time()
    df_user['has_desc'] = df['user_description'].apply(lambda x: x != None)
    dur = time.time() - start
    print('has_desc:', dur)

    start = time.time()
    df_user['has_url'] = df['user_url'].apply(lambda x: x != None)
    dur = time.time() - start
    print('has_url:', dur)

    return df_user

# start = time.time()
df = tweet_json_to_df(dataset_path)

# build feature table for different feature categories
msg_feat_df = msg_feature_df(df)
usr_feat_df = usr_feature_df(df)

retweet_cnt = retweet_cnt(df[['id_str']])
print(len(msg_feat_df),len(usr_feat_df),len(retweet_cnt))

print('\nValue Frequencies:')
print(retweet_cnt['retweet_count'].value_counts())

df = pd.concat([msg_feat_df, usr_feat_df], axis=1)
df = pd.concat([df, retweet_cnt], axis=1)

# missing = pd.Series(df.index[pd.isnull(df['retweet_count'])])
# print(missing)

df.to_csv(file_name+'_feature_set.csv',sep=';',index = True)
print('\nSaved to ...\\'+file_name+'_feature_set.csv')

# pd.DataFrame.to_csv(df,path_or_buf = file_name+'_feature_set.csv',sep=';')