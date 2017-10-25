'''
The script builds a word frequency list out of all the words in a text corpus.
The input file must be a JSON containing Tweet Objects. The json is converted to a 
pandas dataframe and the TEXT column is extracted for counting frequencies. All words 
are converted to lower case. String shorter than one character are ignored.
'''

from tweet_tk.tweets_to_df import tweet_json_to_df
from easygui import fileopenbox
from os import getcwd
from nltk import word_tokenize
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

file_path = fileopenbox(default=getcwd())

try:
    df = tweet_json_to_df(file_path)
except Exception as e:
    print('Error:',e)
    print('Unsuitable file type. Please select JSON with collected Tweets. The Dataframe must contain a "Text" Column')
    quit()


all_words = dict()

for line in df['text']:
    words = [str.lower(w) for w in word_tokenize(line) if w not in stopWords
                                            and len(w)>1
                                            and '//t.co' not in w]
    for w in words:
        if w in all_words.keys():
            all_words[w] += str(line).lower().count(w)
        else:
            all_words[w] = str(line).lower().count(w)

for w in sorted(all_words, key=all_words.get,reverse=True):
    print(all_words[w],w)
