from tweets_to_df import tweet_json_to_df
from emoticons_parser import emoticons_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from nltk import pos_tag
import nltk
import pandas as pd
import time

from textblob import Blobber
from textblob_aptagger import PerceptronTagger
tb = Blobber(pos_tagger=PerceptronTagger())

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
    # tagged_words = pos_tag(word_list)
    # start = time.time()
    tagged_words = tb(text)
    # dur = time.time() - start
    # print('tokenize:', dur)

    # start = time.time()
    # dc = dict(tagged_words.tags)
    # dur = time.time() - start
    # print('convert to dict:', dur)

    # start = time.time()

    dc = [x[1] for x in tagged_words.pos_tags]

    # dur = time.time() - start
    # print('build pos list:', dur)

    # start = time.time()
    ret = ('PRP' in dc)
    # dur = time.time() - start
    # print('scanning dict:', dur)

    return ret

def count_upper(text):
    return sum(1 for c in text if c.isupper())/len(text)

def build_feature_df(df):

    start = time.time()
    df['words'] = df['text'].apply(lambda x : word_tokenize(x))
    dur = time.time() - start
    print('tokenize words:',dur)

    start = time.time()
    stop_words = set(stopwords.words('english'))
    df['words'] = df['words'].apply(lambda x : [w for w in x if w.lower() not in stop_words])
    dur = time.time() - start
    print('filter out stop words:',dur)

    start = time.time()
    df['len_characters'] = df['text'].apply(lambda x : len(x))
    dur = time.time() - start
    print('len_characters:', dur)

    start = time.time()
    df['num_words'] = df['words'].apply(lambda x : len(x))
    dur = time.time() - start
    print('num_words:', dur)

    start = time.time()
    df['has_question_mark'] = df['text'].apply(lambda x : x.find('?') != -1)
    dur = time.time() - start
    print('has_question_mark:', dur)

    start = time.time()
    df['has_exclamation_mark'] = df['text'].apply(lambda x : x.find('!') != -1)
    dur = time.time() - start
    print('has_exclamation_mark:', dur)

    start = time.time()
    df['has_multi_quest_exclam'] = df['text'].apply(lambda x : (x.count('?') > 1 or x.count('!') > 1))
    dur = time.time() - start
    print('has_multi_quest_exclam:', dur)

    start = time.time()
    df['emotji_sent_score'] = df['text'].apply(lambda x : emoticons_score(x))
    dur = time.time() - start
    print('emotji_sent_score:', dur)

    start = time.time()
    df['has_pronoun'] = df['text'].apply(lambda x : has_pronoun(x))
    dur = time.time() - start
    print('has_pronoun:', dur)

    start = time.time()
    df['count_upper'] = df['text'].apply(lambda x : count_upper(x))
    dur = time.time() - start
    print('count_upper:', dur)

    df = df[['words',
        'len_characters','num_words','has_question_mark',
        'has_exclamation_mark','has_multi_quest_exclam','emotji_sent_score',
        'has_pronoun','count_upper']]

    return df

start = time.time()
df = tweet_json_to_df('amazon_dataset.json')
dur = time.time() - start
print('tweet_json_to_df:', dur)

start = time.time()
df = build_feature_df(df)
dur = time.time() - start
print('\nbuild_feature_df:', dur)

pd.DataFrame.to_csv(df,path_or_buf='feature_set.csv',sep=';')
# print(df.head())