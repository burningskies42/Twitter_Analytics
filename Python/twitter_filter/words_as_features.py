import nltk
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.classify import ClassifierI

from string import punctuation

import random
from statistics import mode

import pickle
from re import MULTILINE,sub
from os import getcwd

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


import pandas as pd
import numpy as np

from tweet_tk.fetch_tweets import fetch_tweets_by_ids

stop_words = set(stopwords.words('english'))
punctuation = set([w for w in punctuation])
stop_words_punctuation = set.union(stop_words,punctuation)

class VoteClassifier(ClassifierI):
   def __init__(self, *classifiers):
      self._classifiers = classifiers

   def classify(self, features):
      votes = []
      for c in self._classifiers:
         v = c.classify(features)
         votes.append(v)
      return mode(votes)

   def confidence(self, features):
      votes = []
      for c in self._classifiers:
         v = c.classify(features)
         votes.append(v)

      choice_votes = votes.count(mode(votes))
      conf = choice_votes / len(votes)
      return conf

def build_word_list(sentence):
   sentence = clear_urls(sentence)
   sentence = sentence.replace('\n','')
   sentence = sentence.replace('\r', '')

   for c in punctuation:
      sentence = sentence.replace(c,'')


   words = word_tokenize(sentence)
   words = [w for w in words if w.lower() not in stop_words]

   return words

def txt_lbl(number):
   if number == 1:
      return 'news'
   elif number == 2:
      return 'spam'


def clear_urls(text):
   clear_text = sub(r'https?:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text, flags=MULTILINE)
   clear_text = clear_text.replace('\n', ' ')
   return clear_text

# List of tuples: each tuple contains a list of words(tokenized sentence) + category('pos' or 'neg')
label_df = pd.DataFrame.from_csv('labels/combo.csv',sep=';')
label_df.index = label_df.index.map(str)
label_df['label'] = label_df['label'].apply(lambda x : txt_lbl(x) )
ids = label_df.index.values

print('Dataset size is',len(ids),'tweets')
print('------------------------------------')
df = fetch_tweets_by_ids(ids)[['text']]
df['text'] = df['text'].apply(clear_urls)
df['text'] = df['text'].apply(lambda x : build_word_list(x) )

df = label_df.join(df)
df.dropna(inplace=True)
df.to_csv('test.test',sep=';')

# import chardet
# with open('test.test', 'rb') as f:
#    result = chardet.detect(f.read())  # or readline if the file is large
#    print(result)

# df = pd.read_csv('test.test',sep=';', encoding=result['encoding'])
# # df = pd.DataFrame().from_csv('test.test',sep=';')

documents = [(row['text'],row['label']) for ind,row in df.iterrows()]

# for d in documents:
#    print(d)
# quit()

random.shuffle(documents)
class_ratio = int(len(df[df['label']=='spam'])/len(df[df['label']=='news']))
# print(class_ratio)
all_words = {}

# for w in movie_reviews.words():
#    all_words.append(w.lower())
#
# all_words = nltk.FreqDist(all_words)
for tweet in documents:
   if tweet[1] == 'news' :
      mult = class_ratio
   else:
      mult = 1

   for word in tweet[0]:
      if word.lower() in all_words.keys():
         all_words[word.lower()] += 1*mult
      else:
         all_words[word.lower()] = 1*mult

# for word,cnt in sorted(all_words.items(), key=lambda x:x[1],reverse=True)[:10]:
#    print(word,cnt)
# quit()

word_features = list(all_words.keys())[:3000]
# print(word_features)
# quit()

def find_features(document):
   words = set(document)
   features = {}
   for w in word_features:
      features[w] = (w in words)

   return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]

# for f in featuresets:
#    if f[1]=='news':
#       print(f)
# quit()

training_set_size = int(len(featuresets)*0.6)
print('training set size',training_set_size)
print('testing set size',len(featuresets) - training_set_size)


training_set = featuresets[:training_set_size]
testing_set = featuresets[training_set_size:]

classifier = nltk.NaiveBayesClassifier.train(training_set)

with open(getcwd()+"\\classifiers\\words_as_features\\naivebayes.pickle", "wb") as classifier_f:
   pickle.dump(classifier,classifier_f)
   classifier_f.close()

print("Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set)) * 100)
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
with open(getcwd()+"\\classifiers\\words_as_features\\MNB_classifier.pickle", "wb") as classifier_f:
   pickle.dump(MNB_classifier,classifier_f)
   classifier_f.close()
print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set)) * 100)


BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
with open(getcwd()+"\\classifiers\\words_as_features\\BernoulliNB_classifier.pickle", "wb") as classifier_f:
   pickle.dump(BernoulliNB_classifier,classifier_f)
   classifier_f.close()
print("BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
with open(getcwd()+"\\classifiers\\words_as_features\\LogisticRegression_classifier.pickle", "wb") as classifier_f:
   pickle.dump(LogisticRegression_classifier,classifier_f)
   classifier_f.close()
print("LogisticRegression_classifier accuracy percent:",
      (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
with open(getcwd()+"\\classifiers\\words_as_features\\SGDClassifier_classifier.pickle", "wb") as classifier_f:
   pickle.dump(SGDClassifier_classifier,classifier_f)
   classifier_f.close()
print("SGDClassifier_classifier accuracy percent:",
      (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
with open(getcwd()+"\\classifiers\\words_as_features\\SVC_classifier.pickle", "wb") as classifier_f:
   pickle.dump(SVC_classifier,classifier_f)
   classifier_f.close()
print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
with open(getcwd()+"\\classifiers\\words_as_features\\LinearSVC_classifier.pickle", "wb") as classifier_f:
   pickle.dump(LinearSVC_classifier,classifier_f)
   classifier_f.close()
print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

# SVR_classifier = SklearnClassifier(SVR())
# SVR_classifier.train(training_set)
# print("SVR_classifier accuracy percent:", (nltk.classify.accuracy(SVR_classifier, testing_set)) * 100)

MLP_Classifier = SklearnClassifier(MLPClassifier(alpha=1))
MLP_Classifier.train(training_set)
print("MLP_Classifier accuracy percent:", (nltk.classify.accuracy(MLP_Classifier, testing_set)) * 100)


RandomForest_Classifier = SklearnClassifier(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))
RandomForest_Classifier.train(training_set)
print("RandomForest_Classifier accuracy percent:", (nltk.classify.accuracy(RandomForest_Classifier, testing_set)) * 100)

AdaBoost_Classifier = SklearnClassifier(AdaBoostClassifier())
AdaBoost_Classifier.train(np.array(training_set))
print("AdaBoost_Classifier accuracy percent:", (nltk.classify.accuracy(AdaBoost_Classifier, testing_set)) * 100)



voted_classifier = VoteClassifier(classifier,
                                  # SVR_classifier,
                                  MLP_Classifier,
                                  RandomForest_Classifier,
                                  # QDA_Classifier,
                                  AdaBoost_Classifier,
                                  LinearSVC_classifier,
                                  SGDClassifier_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

print("voted_classifier accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)

# print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",
#       voted_classifier.confidence(testing_set[0][0]) * 100)
# print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",
#       voted_classifier.confidence(testing_set[1][0]) * 100)
# print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",
#       voted_classifier.confidence(testing_set[2][0]) * 100)
# print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",
#       voted_classifier.confidence(testing_set[3][0]) * 100)
# print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",
#       voted_classifier.confidence(testing_set[4][0]) * 100)
# print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",
#       voted_classifier.confidence(testing_set[5][0]) * 100)