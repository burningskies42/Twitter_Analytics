from words_as_features import WordsClassifier
# from tweet_tk import

def train_combo():
   wc = WordsClassifier('train')
   featuresets = [(wc.find_features(rev, wc.word_features), category) for (rev, category) in wc.documents]
   print(featuresets)
# WordsClassifier('load')
train_combo()