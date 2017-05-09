from tweet_tk.tweets_to_df import tweet_json_to_df
from easygui import fileopenbox
from os import getcwd
from nltk import word_tokenize
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))


file_path = fileopenbox(default=getcwd())
df = tweet_json_to_df(file_path)

all_words = dict()

for line in df['text']:
    words = [w for w in word_tokenize(line) if w not in stopWords and len(w)>1]
    for w in words:
        if w in all_words.keys():
            all_words[w] += str(line).count(w)
        else:
            all_words[w] = str(line).count(w)

for w in sorted(all_words, key=all_words.get,reverse=True):
    print(all_words[w],w)
