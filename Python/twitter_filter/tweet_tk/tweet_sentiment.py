# run classifier trainer first, File: TweeterTK.sentiment_train.py
from random import shuffle
from pickle import load
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from os import getcwd


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

f_path = getcwd() + '\\tweet_tk\\pickled_algos\\'
documents_f = open(f_path + "documents.pickle", "rb")
documents = load(documents_f)
documents_f.close()

word_features5k_f = open(f_path +"word_features5k.pickle", "rb")
word_features = load(word_features5k_f)
word_features5k_f.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

featuresets_f = open(f_path + "featuresets.pickle", "rb")
featuresets = load(featuresets_f)
featuresets_f.close()

shuffle(featuresets)


open_file = open(f_path + "originalnaivebayes5k.pickle", "rb")
classifier = load(open_file)
open_file.close()


open_file = open(f_path + "MNB_classifier5k.pickle", "rb")
MNB_classifier = load(open_file)
open_file.close()

open_file = open(f_path + "BernoulliNB_classifier5k.pickle", "rb")
BernoulliNB_classifier = load(open_file)
open_file.close()

open_file = open(f_path + "LogisticRegression_classifier5k.pickle", "rb")
LogisticRegression_classifier = load(open_file)
open_file.close()

open_file = open(f_path + "LinearSVC_classifier5k.pickle", "rb")
LinearSVC_classifier = load(open_file)
open_file.close()

open_file = open(f_path + "SGDC_classifier5k.pickle", "rb")
SGDC_classifier = load(open_file)
open_file.close()

voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)

def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)