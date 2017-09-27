from os import getcwd
import pickle
import pandas as pd

with open("C:/Users/Leon/Documents/Masterarbeit/Python/twitter_filter/classifiers/words_as_features/all_Words.pickle", 'rb') as fid:
   word_features = pickle.load(fid)
   fid.close()

se = pd.Series(sorted(word_features.items(), key=lambda x: x[1], reverse=True))
se.to_csv('all_words.csv',sep=';')
print(se)