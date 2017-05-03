# my modules
from tweets_to_df import tweet_json_to_df
from emoticons_parser import emoticons_score
from retweet_fetcher import retweet_cnt
from url_parser import most_pop_urls
from tweet_sentiment import sentiment

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
import re
from difflib import SequenceMatcher
from pickle import dump, load

def similar(df):
    sent_df = df.to_frame()
    sent_df['similiarity'] = ''

    for i,sentence in sent_df.iterrows():
        max_ratio = 0
        for j,another in sent_df.iterrows():

            # id different tweets, compare to max
            if i != j:
                simil= SequenceMatcher(None, sentence, another).ratio()
                if simil == 1:
                    sent_df['similiarity'][i] = 1
                    # sent_df['similiarity'][j] = 1
                    break

                elif simil > max_ratio:
                    max_ratio = simil

        sent_df['similiarity'][i] = max_ratio
    return sent_df['similiarity']

# pip install -U git+https://github.com/sloria/textblob-aptagger.git@dev
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

def clear_urls(text):
    clear_text = re.sub(r'https?:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text,flags=re.MULTILINE)
    return clear_text

def add_suspects(set_of_new_ids):
    suspect_file = open('bot_suspects\\bot_suspects.pickle','rb')
    old_suspects = set(load(suspect_file))
    print('------------------\n'+str(len(old_suspects))
          ,'old suspects in list')
    suspect_file.close()

    unique_suspects = set_of_new_ids.difference(old_suspects)
    print(len(set_of_new_ids),'bot suspects in current dataset, of them',
          len(unique_suspects), 'are new')

    updated_suspects = set(old_suspects).union(unique_suspects)
    # updated_suspects = set_of_new_ids
    print('updated suspect list length is',
          len(updated_suspects),
          '\n------------------')
    suspect_file = open('bot_suspects\\bot_suspects.pickle', 'wb')
    dump(updated_suspects, suspect_file)
    suspect_file.close()


def msg_feature_df(df):
    df_msg = pd.DataFrame()
    df_msg['id'] = df.index
    df_msg.set_index('id', inplace=True)

    start = time.time()
    df_msg['words'] = df['text'].apply(lambda x : word_tokenize(x))
    dur = time.time() - start
    print('tokenize words:',dur)

    start = time.time()
    df_msg['words_no_url'] = df['text'].apply(lambda x : clear_urls(x))
    dur = time.time() - start
    print('clear urls:', dur)

    start = time.time()
    df_msg['duplicate'] = df_msg['words_no_url'].duplicated(keep=False)
    dur = time.time() - start
    print('duplicates:', dur)

    start = time.time()
    bot_suspects = set(df['user_id'][df_msg['duplicate']==True])
    add_suspects(bot_suspects)
    dur = time.time() - start
    print('add_suspects:', dur)


    # start = time.time()
    # df_msg['similiarity'] = similar(df_msg['words_no_url'])
    # print(similar(df_msg['words_no_url']))
    # dur = time.time() - start
    # print('similiarity:', dur)

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

    start = time.time()
    sentm = df['text'].apply(lambda x: sentiment(x))
    sentm = sentm.apply(pd.Series)
    sentm.columns = ['class', 'conf']
    df_msg['senitment'] = sentm['class']
    df_msg['senitment_conf'] = sentm['conf']
    dur = time.time() - start
    print('senitment:', dur)

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

df.to_csv('feature_sets\\'+file_name+'_feature_set.csv',sep=';',index = True)
print('\nSaved to ...\\'+file_name+'_feature_set.csv')

# pd.DataFrame.to_csv(df,path_or_buf = file_name+'_feature_set.csv',sep=';')