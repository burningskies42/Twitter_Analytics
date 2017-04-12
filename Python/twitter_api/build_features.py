from tweets2df import *
import re

# the class will produce and contain all features
class tweet_fetures():
    def __init__(self,tweet):
        # msg. features
        self.text = tweet['text']
        self.words = self.text.split()
        self.len_characters = len(self.text)
        self.len_words = len(self.words)
        self.has_question_mark = self.text.find('?') != -1
        self.has_exclamation_mark = self.text.find('!') != -1

        self.has_multi_quest_exclam = self.text.count('?') > 1 or self.text.count('!') > 1


df = tweet_pickle_to_df('amazon_dataset_1.pickle')
first_entry = df.iloc[77]

try_class = tweet_fetures(first_entry)
print(try_class.text)
