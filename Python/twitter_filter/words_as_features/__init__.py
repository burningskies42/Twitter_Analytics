from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.classify import ClassifierI
from nltk import NaiveBayesClassifier,classify
from nltk import pos_tag

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC       #, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from string import punctuation
from statistics import mode
import random
import pickle
from re import MULTILINE,sub
from os import getcwd,get_exec_path
import pandas as pd
import numpy as np

from tweet_tk.fetch_tweets import fetch_tweets_by_ids

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

class WordsClassifier():
   def __init__(self,load_train='load',pth = ''):
      assert load_train in ('load','train','')

      self.voter = None
      self.all_words = {}
      self.stop_words = set(stopwords.words('english'))
      self.punct = punctuation
      self.punct = set([w for w in self.punct])
      self.stop_words_punctuation = set.union(self.stop_words, self.punct)
      self.lmtzr = WordNetLemmatizer()
      self.allowed_word_types = ["N", "J", "R", "V"]
      self.word_features = []

      if load_train == 'load':
         self.load_classifier()
      elif load_train == 'train':
         self.train(pth=pth)

   def build_word_list(self, sentence):
      sentence = self.clear_urls(sentence)
      sentence = sentence.replace('\n', '')
      sentence = sentence.replace('\r', '')

      for c in punctuation:
         sentence = sentence.replace(c, '')

      words = word_tokenize(sentence)

      # print(words)
      words = [w.lower() for w in words if w.lower() not in self.stop_words]

      lms_words = list(map(lambda x: self.lmtzr.lemmatize(x), words))
      words = pos_tag(lms_words)

      # words = [w[0] for w in words if w[1][0] in allowed_word_types]
      words = [w[0] for w in words if w[1][0] in self.allowed_word_types]
      # print(words)
      # print('~~~~~~~~~')
      return words

   def txt_lbl(self, number):
      if number == 1:
         return 'news'
      elif number == 2:
         return 'spam'

   def clear_urls(self, text):
      clear_text = sub(r'https?:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text, flags=MULTILINE)
      clear_text = clear_text.replace('\n', ' ')
      return clear_text

   def find_features(self, document, word_features):
      words = set(document)
      features = {}
      for w in word_features:
         features[w] = (w in words)

      return features

   def str_to_featureset(self, str):
      str = self.clear_urls(str)
      str = self.build_word_list(str)
      featureset = self.find_features(str, self.word_features)

      return featureset

   def fetch_tweets(self,pth,with_print=True):

      # List of tuples: each tuple contains a list of words(tokenized sentence) + category('pos' or 'neg')
      label_df = pd.DataFrame.from_csv(pth,sep=';')
      label_df.index = label_df.index.map(str)
      label_df['label'] = label_df['label'].apply(lambda x : self.txt_lbl(x) )
      ids = label_df.index.values

      if with_print: print('Dataset size is',len(ids),'tweets')
      if with_print: print('------------------------------------')
      df = fetch_tweets_by_ids(ids)[['text']]
      df['text'] = df['text'].apply(self.clear_urls)
      df['text'] = df['text'].apply(lambda x : self.build_word_list(x) )

      df = label_df.join(df)
      df.dropna(inplace=True)
      # df.to_csv('test.test',sep=';')

      self.documents = [(row['text'],row['label']) for ind,row in df.iterrows()]
      # print(self.documents)

      random.shuffle(self.documents)
      self.class_ratio = int(len(df[df['label']=='spam'])/len(df[df['label']=='news']))

   def train(self,pth='C:/Users/Leon/Documents/Masterarbeit/Python/twitter_filter/labels/combo.csv',with_print=True):
      print(pth)
      self.fetch_tweets(with_print=with_print,pth=pth)

      for tweet in self.documents:
         if tweet[1] == 'news' :
            mult = self.class_ratio
         else:
            mult = 1

         for word in tweet[0]:
            if word.lower() in self.all_words.keys():
               self.all_words[word.lower()] += 1*mult
            else:
               self.all_words[word.lower()] = 1*mult

      # Get the 5000 most popular words
      self.word_features=sorted(self.all_words.items(), key=lambda x:x[1],reverse=True)[:5000]
      self.word_features = [w[0] for w in self.word_features]

      featuresets = [(self.find_features(rev,self.word_features), category) for (rev, category) in self.documents]

      with open(getcwd()+"\\classifiers\\words_as_features\\Words.pickle", "wb") as fid:
         pickle.dump(self.word_features, fid)
         fid.close()

      training_set_size = int(len(featuresets)*0.6)
      if with_print: print('training set size',training_set_size)
      if with_print: print('testing set size',len(featuresets) - training_set_size)


      training_set = featuresets[:training_set_size]
      testing_set = featuresets[training_set_size:]

      Naivebayes_classifier = NaiveBayesClassifier.train(training_set)
      with open(getcwd()+"\\classifiers\\words_as_features\\Naivebayes_classifier.pickle", "wb") as classifier_f:
         pickle.dump(Naivebayes_classifier,classifier_f)
         classifier_f.close()

      if with_print:
         print("Original Naive Bayes Algo accuracy percent:",
               (classify.accuracy(Naivebayes_classifier, testing_set)) * 100)
         Naivebayes_classifier.show_most_informative_features(15)

      MNB_classifier = SklearnClassifier(MultinomialNB())
      MNB_classifier.train(training_set)
      with open(getcwd()+"\\classifiers\\words_as_features\\MNB_classifier.pickle", "wb") as classifier_f:
         pickle.dump(MNB_classifier,classifier_f)
         classifier_f.close()
      if with_print:
         print("MNB_classifier accuracy percent:",
               (classify.accuracy(MNB_classifier, testing_set)) * 100)


      BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
      BernoulliNB_classifier.train(training_set)
      with open(getcwd()+"\\classifiers\\words_as_features\\BernoulliNB_classifier.pickle", "wb") as classifier_f:
         pickle.dump(BernoulliNB_classifier,classifier_f)
         classifier_f.close()
      if with_print: print("BernoulliNB_classifier accuracy percent:", (classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

      LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
      LogisticRegression_classifier.train(training_set)
      with open(getcwd()+"\\classifiers\\words_as_features\\LogisticRegression_classifier.pickle", "wb") as classifier_f:
         pickle.dump(LogisticRegression_classifier,classifier_f)
         classifier_f.close()
      if with_print:
         print("LogisticRegression_classifier accuracy percent:",
               (classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

      SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
      SGDClassifier_classifier.train(training_set)
      with open(getcwd()+"\\classifiers\\words_as_features\\SGDClassifier_classifier.pickle", "wb") as classifier_f:
         pickle.dump(SGDClassifier_classifier,classifier_f)
         classifier_f.close()
      if with_print:
         print("SGDClassifier_classifier accuracy percent:",
               (classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)

      SVC_classifier = SklearnClassifier(SVC())
      SVC_classifier.train(training_set)
      with open(getcwd()+"\\classifiers\\words_as_features\\SVC_classifier.pickle", "wb") as classifier_f:
         pickle.dump(SVC_classifier,classifier_f)
         classifier_f.close()
      if with_print:
         print("SVC_classifier accuracy percent:",
               (classify.accuracy(SVC_classifier, testing_set))*100)

      LinearSVC_classifier = SklearnClassifier(LinearSVC())
      LinearSVC_classifier.train(training_set)
      with open(getcwd()+"\\classifiers\\words_as_features\\LinearSVC_classifier.pickle", "wb") as classifier_f:
         pickle.dump(LinearSVC_classifier,classifier_f)
         classifier_f.close()
      if with_print:
         print("LinearSVC_classifier accuracy percent:",
               (classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

      # SVR_classifier = SklearnClassifier(SVR())
      # SVR_classifier.train(training_set)
      # print("SVR_classifier accuracy percent:", (nltk.classify.accuracy(SVR_classifier, testing_set)) * 100)

      MLP_Classifier = SklearnClassifier(MLPClassifier(alpha=1))
      MLP_Classifier.train(training_set)
      if with_print:
         print("MLP_Classifier accuracy percent:",
               (classify.accuracy(MLP_Classifier, testing_set)) * 100)


      RandomForest_Classifier = SklearnClassifier(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))
      RandomForest_Classifier.train(training_set)
      if with_print:
         print("RandomForest_Classifier accuracy percent:",
               (classify.accuracy(RandomForest_Classifier, testing_set)) * 100)

      AdaBoost_Classifier = SklearnClassifier(AdaBoostClassifier())
      AdaBoost_Classifier.train(np.array(training_set))
      if with_print:
         print("AdaBoost_Classifier accuracy percent:",
               (classify.accuracy(AdaBoost_Classifier, testing_set)) * 100)

      self.voted_classifier = VoteClassifier(Naivebayes_classifier,
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

      if with_print: print("voted_classifier accuracy percent:", (classify.accuracy(self.voted_classifier, testing_set)) * 100)

   def load_classifier(self):
      with open(getcwd() + "\\classifiers\\words_as_features\\BernoulliNB_classifier.pickle",'rb') as fid:
         BernoulliNB_classifier = pickle.load(fid)
         fid.close()

      with open(getcwd() + "\\classifiers\\words_as_features\\LinearSVC_classifier.pickle",'rb') as fid:
         LinearSVC_classifier = pickle.load(fid)
         fid.close()

      with open(getcwd() + "\\classifiers\\words_as_features\\LogisticRegression_classifier.pickle", 'rb') as fid:
         LogisticRegression_classifier = pickle.load(fid)
         fid.close()

      with open(getcwd() + "\\classifiers\\words_as_features\\MNB_classifier.pickle", 'rb') as fid:
         MNB_classifier = pickle.load(fid)
         fid.close()

      with open(getcwd() + "\\classifiers\\words_as_features\\Naivebayes_classifier.pickle", 'rb') as fid:
         Naivebayes_classifier = pickle.load(fid)
         fid.close()

      with open(getcwd() + "\\classifiers\\words_as_features\\SGDClassifier_classifier.pickle", 'rb') as fid:
         SGDClassifier_classifier = pickle.load(fid)
         fid.close()

      with open(getcwd() + "\\classifiers\\words_as_features\\SVC_classifier.pickle", 'rb') as fid:
         SVC_classifier = pickle.load(fid)
         fid.close()

      with open(getcwd() + "\\classifiers\\words_as_features\\words.pickle", 'rb') as fid:
         self.word_features = pickle.load(fid)
         fid.close()

      self.voter = VoteClassifier(
         BernoulliNB_classifier,
         LinearSVC_classifier,
         LogisticRegression_classifier,
         MNB_classifier,
         Naivebayes_classifier,
         SGDClassifier_classifier,
         SVC_classifier
      )

