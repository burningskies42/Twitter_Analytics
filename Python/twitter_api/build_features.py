from tweets_to_df import tweet_json_to_df
from emoticons_parser import emoticons_score
import pandas as pd

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

df = tweet_json_to_df('amazon_dataset.json')

df['words'] = df['text'].apply(lambda x : x.split())
df['len_characters'] = df['text'].apply(lambda x : len(x))
df['num_words'] = df['words'].apply(lambda x : len(x))
df['has_question_mark'] = df['text'].apply(lambda x : x.find('?') != -1)
df['has_exclamation_mark'] = df['text'].apply(lambda x : x.find('!') != -1)
df['has_multi_quest_exclam'] = df['text'].apply(lambda x : (x.count('?') > 1 or x.count('!') > 1))
df['emotji_sent_score'] = df['text'].apply(lambda x : emoticons_score(x))

df = df[['words','len_characters','num_words','has_question_mark',
    'has_exclamation_mark','has_multi_quest_exclam','emotji_sent_score']]

print(df.head(),)