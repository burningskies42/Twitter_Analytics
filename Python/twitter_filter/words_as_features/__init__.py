from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.classify import ClassifierI
from nltk import NaiveBayesClassifier,classify
from nltk import pos_tag

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.linear_model import LinearRegression
from sklearn import model_selection,preprocessing
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import roc_auc_score,roc_curve,auc
import matplotlib.pyplot as plt
from sklearn import tree

import pydot
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


from sklearn.metrics import cohen_kappa_score,confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import scipy.stats as stats
import os
import math




from string import punctuation
from statistics import mode
import random
import pickle
from re import MULTILINE,sub
from os import getcwd,get_exec_path
import pandas as pd
import time
import datetime
# import matplotlib.pyplot as plt

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

class Kappa():
   '''
   creates a true/false - negative/positive matrix from calculating a Kappa value
   @:param(classifier) - classifier which was trained using a learning algorithm
   '''
   def __init__(self,classifier,testing_set):
      self.kappa_matrix = {'news':{'true':0,'false':0},
                           'spam':{'true':0,'false':0}}
      self.clf = classifier

      # Get classifier name
      try:
         self.clf_name = self.clf._clf.__class__.__name__
      except Exception as e:
         self.clf_name = self.clf.__class__.__name__

      self.clf_name = self.clf_name.replace('Classifier','')

      y_true = [1 if line[1]=='news' else 0 for line in testing_set]

      Xs = [line[0] for line in testing_set]

      y_predictions = self.clf.classify_many(Xs)
      y_predictions = [1 if line == 'news' else 0 for line in y_predictions]

      self.rauc_score = round(roc_auc_score(y_true,y_predictions) * 100, 2)

      # Count False Pos/Neg and True Pos/Neg
      # Where Pos = News and Neg = Spam
      for fs in testing_set:

         # true classification is 'news'
         alg_class = self.clf.classify(fs[0])

         if fs[1] == 'news':
            # algo classified correctly (a)
            if alg_class == 'news':
               self.kappa_matrix['news']['true'] += 1

            # algo misclassified 'news' as 'spam' (b)
            else:
               self.kappa_matrix['spam']['false'] += 1

         # true classification is 'spam'
         elif fs[1] == 'spam':
            # # algo classified correctly (d)
            if alg_class == 'spam':
               self.kappa_matrix['spam']['true'] += 1

            # algo misclassified 'spam' as 'news' (c)
            else:
               self.kappa_matrix['news']['false'] +=1

      # Kappa Calculation
      # https://en.wikipedia.org/wiki/Cohen%27s_kappa
      a = self.kappa_matrix['news']['true']  # True Positive
      b = self.kappa_matrix['spam']['false'] # False Negative
      c = self.kappa_matrix['news']['false'] # False Positive
      d = self.kappa_matrix['spam']['true']  # True Negative
      abcd = a+b+c+d
      p_zero = (a+d)/abcd
      p_news = ((a + b) / abcd) * ((a + c) / abcd)
      p_spam = ((c + d) / abcd) * ((b + d) / abcd)
      p_e = p_news + p_spam

      self.kappa_value = 100*(p_zero - p_e) / (1 - p_e)
      self.kappa_value = round(self.kappa_value,2)
      self.accuracy = round((classify.accuracy(classifier, testing_set)) * 100, 2)

      '''
      TPR       = TP/(TP+FN) = TP/P 
      FPR       = FP/(FP+TN) = FP/N  
      Precision = TP/(TP+FP) 
      Recall    = TP/(TP+FN)
      '''

      self.prec_recall = pd.DataFrame(columns=['Name','True','False','TPR','FPR','Prec','Recall','F1','Kappa','Accuracy'],index=['News','Not-News'])
      self.news = {}
      self.news['tpr']    = round(a / (a + b), 2) if a+b != 0 else 0
      self.news['fpr']    = round(c / (c + d), 2) if c+d != 0 else 0
      self.news['prec']   = round(a / (a + c), 2) if a+c != 0 else 0
      self.news['recall'] = round(a / (a + b), 2) if a+b != 0 else 0
      self.news['F1'] = (2*self.news['prec']*self.news['recall'])/(self.news['prec']+self.news['recall']) if self.news['prec']+self.news['recall'] != 0 else 0
      self.news['F1'] = round(self.news['F1'],2)
      '''------------------------------------------------------------------------------------------'''
      s_news = pd.Series({'Name':self.clf_name,
                          'True':self.kappa_matrix['news']['true'],
                          'False':self.kappa_matrix['news']['false'],
                          'TPR':self.news['tpr'],
                          'FPR':self.news['fpr'],
                          'Prec':self.news['prec'],
                          'Recall':self.news['recall'],
                          'F1': self.news['F1'],
                          'Kappa':self.kappa_value,
                          'Accuracy': self.accuracy
                          })
      self.prec_recall.loc['News'] = s_news

      self.spam = {}
      self.spam['tpr']    = round(d / (c + d), 2) if c+d != 0 else 0
      self.spam['fpr']    = round(b / (a + b), 2) if a+b != 0 else 0
      self.spam['prec']   = round(d / (b + d), 2) if b+d != 0 else 0
      self.spam['recall'] = round(d / (c + d), 2) if c+d != 0 else 0
      self.spam['F1'] = (2 * self.spam['prec'] * self.spam['recall']) / (self.spam['prec'] + self.spam['recall']) if self.spam['prec']+self.spam['recall'] != 0 else 0
      self.spam['F1'] = round(self.spam['F1'], 2)
      '''------------------------------------------------------------------------------------------'''
      s_spam = pd.Series({'Name':self.clf_name,
                          'True': self.kappa_matrix['spam']['true'],
                          'False': self.kappa_matrix['spam']['false'],
                          'TPR':self.spam['tpr'],
                          'FPR':self.spam['fpr'],
                          'Prec':self.spam['prec'],
                          'Recall':self.spam['recall'],
                          'F1':self.spam['F1'],
                          'Kappa':self.kappa_value,
                          'Accuracy': self.accuracy
                          })
      self.prec_recall.loc['Not-News'] = s_spam
      self.output = pd.Series({
         'Name': self.clf_name,
         'True_News': self.kappa_matrix['news']['true'],
         'False_News': self.kappa_matrix['news']['false'],
         'True_Spam': self.kappa_matrix['spam']['true'],
         'False_Spam': self.kappa_matrix['spam']['false'],
         'News_TPR': self.news['tpr'],
         'News_FPR': self.news['fpr'],
         'News_Prec': self.news['prec'],
         'News_Recall': self.news['recall'],
         'News_F1': self.news['F1'],
         'Spam_TPR': self.spam['tpr'],
         'Spam_FPR': self.spam['fpr'],
         'Spam_Prec': self.spam['prec'],
         'Spam_Recall': self.spam['recall'],
         'Spam_F1':self.spam['F1'],
         'Kappa': self.kappa_value,
         'Accuracy': self.accuracy,
         'rauc':self.rauc_score,

      })
      self.output.name = self.clf_name
      print()
      print(self.prec_recall[['True','False','TPR','FPR','Prec','Recall','F1']])
      print('Accuracy:', self.accuracy,'%')
      print('Kappa:   ', self.kappa_value,'%')
      print('rAUC:    ', self.rauc_score , '%')

class WordsClassifier():
   '''
   Class must be initialized elsewhere. The class gets a path parameter and outputs
   classification statistics for given dataset.
   @:param(load_train)  - indicates whether new classifier have to be trained 
                          or existing ones loaded from pickles
   @:param(pth)         - loaction of dataset
   @:param(from_server) - If true, all tweets will be fetched from server. Else, tweets will be loaded
                          from last downloaded tweets file
   '''
   def __init__(self,load_train='load',pth = '',from_server=True,num_features=5000,with_trees=False):
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
      self.output_log = pd.DataFrame()
      self.class_ratio = 1
      self.time_stamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.with_trees = with_trees

      self.training_set = None
      self.testining_set = None

      # if load_train == 'load':
      #    self.load_classifier()
      # elif load_train == 'train':
      #    self.train(self.time_stamp,ngrams=ngrams,n_min=n_min,n_max=n_max,num_features=num_features,pth=pth,remove_stopwords=remove_stopwords,fetch_from_server=from_server,with_trees=self.with_trees)

   def build_word_list(self, sentence,remove_stopwords):
      sentence = self.clear_urls(sentence)
      sentence = sentence.replace('\n', '')
      sentence = sentence.replace('\r', '')

      for c in punctuation:
         sentence = sentence.replace(c, '')

      words = sentence

      # Remove stop_words and lemmatize
      if remove_stopwords:
         words = word_tokenize(sentence)
         words = [w.lower() for w in words if w.lower() not in self.stop_words]

         lms_words = list(map(lambda x: self.lmtzr.lemmatize(x), words))
         words = pos_tag(lms_words)

         words = [w[0] for w in words if w[1][0] in self.allowed_word_types]

      return words

   # Instead of list of words retuns a list of n-grams
   def build_ngram_list(self, sentence,n_min,n_max):
      sentence = self.clear_urls(sentence)
      sentence = sentence.replace('\n', '')
      sentence = sentence.replace('\r', '')

      for c in punctuation:
         sentence = sentence.replace(c, '')

      words = ' '.join(sentence.split()).lower()
      word_cnt = len(words.split(' '))

      try:
         ngram_vectorizer = CountVectorizer(ngram_range=(n_min,n_max))
         ngram_vectorizer.fit_transform([words])
         line_ngrams = ngram_vectorizer.get_feature_names()


      except Exception as e:
         print('no ngrams found:',words)
         line_ngrams= [words]

      return line_ngrams

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

   def fetch_tweets(self,pth,remove_stopwords,ngrams,n_min,n_max,with_print):

      # List of tuples: each tuple contains a list of words(tokenized sentence) + category('pos' or 'neg')
      label_df = pd.DataFrame.from_csv(pth,sep=';')
      label_df.index = label_df.index.map(str)
      label_df['label'] = label_df['label'].apply(lambda x : self.txt_lbl(x) )
      ids = label_df.index.values

      if with_print: print('Dataset size is',len(ids),'tweets')
      if with_print: print('------------------------------------')
      df = fetch_tweets_by_ids(ids)[['text']]
      df['text'] = df['text'].apply(self.clear_urls)

      if ngrams:
         df['text'] = df['text'].apply(lambda x: self.build_ngram_list(x,n_min=n_min,n_max=n_max))

      else:
         df['text'] = df['text'].apply(lambda x : self.build_word_list(x,remove_stopwords=remove_stopwords))


      df = label_df.join(df)
      df.dropna(inplace=True)
      df.to_csv(getcwd()+'\\classifiers\\words_as_features\\latest_dataset.csv',sep=';')

      self.documents = [(row['text'],row['label']) for ind,row in df.iterrows()]
      # print(self.documents)

      random.shuffle(self.documents)

      with open(getcwd()+"\\classifiers\\words_as_features\\Documents.pickle", "wb") as fid:
         pickle.dump(self.documents,fid)
         fid.close()
      # self.class_ratio = int(len(df[df['label']=='spam'])/len(df[df['label']=='news']))

   def load_tweets_from_file(self):
      with open(getcwd()+"\\classifiers\\words_as_features\\Documents.pickle", "rb") as fid:
         self.documents = pickle.load(fid)
         fid.close()

   def build_features(self,num_features,feature_selection="tf"):
      self.load_tweets_from_file()

      if feature_selection == "tfidf":
         self.all_words_tf_idf = {}

         num_of_docs = len(self.documents)

         # else:
         for tweet in self.documents:

            for word in set(tweet[0]):

               # Count the frequency of all words
               cnt = [1 if w == word else 0 for w in tweet[0]]
               if word.lower() in self.all_words.keys():
                  self.all_words[word.lower()] += sum(cnt)
               else:
                  self.all_words[word.lower()] = sum(cnt)

               # Count in how many docs a word appears
               # more than one appearance of word in sigle doc, counts as 1
               if word.lower() in self.all_words_tf_idf.keys():
                  self.all_words_tf_idf[word.lower()] += 1
               else:
                  self.all_words_tf_idf[word.lower()] = 1


         self.all_words_tf_idf = {w: round(tf * math.log(num_of_docs / self.all_words_tf_idf[w]),2) for (w, tf) in
                                  self.all_words.items()}
         self.word_features = sorted(self.all_words_tf_idf.items(), key=lambda x: x[1], reverse=True)[10:(num_features + 10)]

      else:
         for tweet in self.documents:

            for word in set(tweet[0]):

               # Count the frequency of all words
               cnt = [1 if w == word else 0 for w in tweet[0]]
               if word.lower() in self.all_words.keys():
                  self.all_words[word.lower()] += sum(cnt)
               else:
                  self.all_words[word.lower()] = sum(cnt)

         with open(getcwd() + "\\classifiers\\words_as_features\\all_Words.pickle", "wb") as fid:
            pickle.dump(self.all_words, fid)
            fid.close()

         self.word_features = sorted(self.all_words.items(), key=lambda x: x[1], reverse=True)[10:(num_features + 10)]


      self.word_features = [w[0] for w in self.word_features]
      self.feature_cnt = len(self.word_features)

      print(self.word_features)

      random.shuffle(self.documents)
      self.featuresets = [(self.find_features(rev, self.word_features), category) for (rev, category) in self.documents]


      with open(getcwd() + "\\classifiers\\words_as_features\\Words.pickle", "wb") as fid:
         pickle.dump(self.word_features, fid)
         # fid.close()
         # quit()

   def train_test_split(self,with_print):
      # Sizes
      training_set_size = int(len(self.featuresets) * 0.7)
      self.training_set = self.featuresets[:training_set_size]
      self.testing_set = self.featuresets[training_set_size:]

      self.sizes_df = pd.DataFrame()
      se = pd.Series({'News': 0, 'Not-News': 0})
      se.name='Training'
      self.sizes_df = self.sizes_df.append(se)
      se.name='Testing'
      self.sizes_df = self.sizes_df.append(se)


      self.sizes_df.loc['Training']['News']     = sum([1 for obs in self.training_set if obs[1] == 'news'])
      self.sizes_df.loc['Training']['Not-News'] = training_set_size - self.sizes_df.loc['Training']['News']
      self.sizes_df.loc['Testing']['News']      = sum([1 for obs in self.testing_set if obs[1] == 'news'])
      self.sizes_df.loc['Testing']['Not-News']  = len(self.testing_set) - self.sizes_df.loc['Testing']['News']

      print(self.sizes_df)

      print('------------------------------------------------------------------------\n', 'Linear Regression:')
      # This split datasets are only used in linear regression

      self.X_train, self.y_train = zip(*self.training_set)
      self.X_train = np.array(pd.DataFrame.from_records(self.X_train))

      # Covert labels to numbers: 1=News, 2=NotNews
      self.y_train = np.array([(i == 'spam') + 1 for i in self.y_train])

      self.X_test, self.y_test = zip(*self.testing_set)
      self.X_test = np.array(pd.DataFrame.from_records(self.X_test))

      self.y_test = np.array([(i == 'spam') + 1 for i in self.y_test])

      LinearRegression_classifier = LinearRegression()
      LinearRegression_classifier.fit(self.X_train, self.y_train)
      R2 = round(LinearRegression_classifier.score(self.X_test, self.y_test), 2)

      news_ys = []
      spam_ys = []


      for x, y in zip(self.X_test, self.y_test):
         y_hat = LinearRegression_classifier.predict(x.reshape(1, -1))[0]
         if y == 1:
            news_ys.append(y_hat)
         else:
            spam_ys.append(y_hat)

      self.test_stats = pd.DataFrame(columns=['News', 'Not-News'])
      self.test_stats.loc['mean'] = pd.Series(
         {'News': round(np.mean(news_ys), 2), 'Not-News': round(np.mean(spam_ys), 2)})
      self.test_stats.loc['median'] = pd.Series(
         {'News': round(np.median(news_ys), 2), 'Not-News': round(np.median(spam_ys), 2)})

      if with_print:
         print(self.test_stats)
         print('Coeffiecient of Determination (R^2):', R2)

   def train(self,with_trees,ngrams,n_min,n_max,with_print):
      # if fetch_from_server:
      #    self.fetch_tweets(with_print=with_print,pth=pth,remove_stopwords=remove_stopwords,ngrams=ngrams,n_min=n_min,n_max=n_max)
      # else:

      # self.train_test_split(with_print)

      # Logistic Regression
      print('------------------------------------------------------------------------\n','Logistic Regression:')
      start_clf_time = time.time()
      LogisticRegression_classifier = SklearnClassifier(LogisticRegression(fit_intercept=True))
      LogisticRegression_classifier.train(self.training_set)

      output = Kappa(LogisticRegression_classifier, self.testing_set).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      with open(getcwd()+"\\classifiers\\words_as_features\\LogisticRegression.pickle", "wb") as classifier_f:
         pickle.dump(LogisticRegression_classifier,classifier_f)
         classifier_f.close()

      print('------------------------------------------------------------------------\n','Naive Bayes:')
      start_clf_time = time.time()
      Naivebayes_classifier = NaiveBayesClassifier.train(self.training_set)

      output = Kappa(Naivebayes_classifier, self.testing_set).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      Naivebayes_classifier.show_most_informative_features(15)

      with open(getcwd()+"\\classifiers\\words_as_features\\Naivebayes_classifier.pickle", "wb") as classifier_f:
         pickle.dump(Naivebayes_classifier,classifier_f)
         classifier_f.close()

      print('------------------------------------------------------------------------\n','Multinomial Naive Bayes:')
      start_clf_time = time.time()
      MNB_classifier = SklearnClassifier(MultinomialNB())
      MNB_classifier.train(self.training_set)

      output = Kappa(MNB_classifier, self.testing_set).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      with open(getcwd()+"\\classifiers\\words_as_features\\MNB_classifier.pickle", "wb") as classifier_f:
         pickle.dump(MNB_classifier,classifier_f)
         classifier_f.close()

      print('------------------------------------------------------------------------\n','Bernoulli Naive Bayes:')
      start_clf_time = time.time()
      BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
      BernoulliNB_classifier.train(self.training_set)

      output = Kappa(BernoulliNB_classifier, self.testing_set).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      with open(getcwd()+"\\classifiers\\words_as_features\\BernoulliNB_classifier.pickle", "wb") as classifier_f:
         pickle.dump(BernoulliNB_classifier,classifier_f)
         classifier_f.close()

      '''
      ================================================================================================================================================
      ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS 
      ================================================================================================================================================
      '''

      print('------------------------------------------------------------------------\n','C-Support Vector Machine:')
      print('======================\n','Linear Kernel')
      start_clf_time = time.time()
      SVC_lin_classifier = SklearnClassifier(SVC(kernel='linear'))
      SVC_lin_classifier.train(self.training_set)

      output = Kappa(SVC_lin_classifier, self.testing_set).output
      output['Kernel'] = 'linear'
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      with open(getcwd()+"\\classifiers\\words_as_features\\SVC_lin.pickle", "wb") as classifier_f:
         pickle.dump(SVC_lin_classifier,classifier_f)
         classifier_f.close()

      print('======================\n', 'Polynomial Kernel')
      start_clf_time = time.time()
      SVC_poly_classifier = SklearnClassifier(SVC(kernel='poly',C=1,gamma=1))
      SVC_poly_classifier.train(self.training_set)

      output = Kappa(SVC_poly_classifier, self.testing_set).output
      output['Kernel'] = 'poly'
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      with open(getcwd() + "\\classifiers\\words_as_features\\SVC_poly.pickle", "wb") as classifier_f:
         pickle.dump(SVC_poly_classifier , classifier_f)
         classifier_f.close()

      # Also default kernel
      print('======================\n', 'Radial Basis Function Kernel')
      start_clf_time = time.time()
      SVC_classifier = SklearnClassifier(SVC(kernel='rbf',gamma=0.1,C=1.38))
      SVC_classifier.train(self.training_set)

      output = Kappa(SVC_classifier, self.testing_set).output
      output['Kernel'] = 'rbf'
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      with open(getcwd() + "\\classifiers\\words_as_features\\SVC_rbf.pickle", "wb") as classifier_f:
         pickle.dump(SVC_classifier, classifier_f)
         classifier_f.close()

      print('======================\n', 'Sigmoid Kernel')
      start_clf_time = time.time()
      SVC_sig_classifier = SklearnClassifier(SVC(kernel='sigmoid',gamma=10))
      SVC_sig_classifier.train(self.training_set)

      output = Kappa(SVC_sig_classifier, self.testing_set).output
      output['Kernel'] = 'sigmoid'
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      with open(getcwd() + "\\classifiers\\words_as_features\\SVC_sigmoid.pickle", "wb") as classifier_f:
         pickle.dump(SVC_sig_classifier, classifier_f)
         classifier_f.close()

      '''
      ================================================================================================================================================
      '''

      print('------------------------------------------------------------------------\n','Stochastic Gradient Descent:')
      start_clf_time = time.time()
      SGD_classifier = SklearnClassifier(SGDClassifier())
      SGD_classifier.train(self.training_set)

      output = Kappa(SGD_classifier, self.testing_set).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      with open(getcwd()+"\\classifiers\\words_as_features\\SGD_classifier.pickle", "wb") as classifier_f:
         pickle.dump(SGD_classifier,classifier_f)
         classifier_f.close()

      print('------------------------------------------------------------------------\n','Multi-layer Perceptron:')
      start_clf_time = time.time()
      MLP_Classifier = SklearnClassifier(MLPClassifier(alpha=1))
      MLP_Classifier.train(self.training_set)

      output = Kappa(MLP_Classifier, self.testing_set).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      with open(getcwd()+"\\classifiers\\words_as_features\\MLP_Classifier.pickle", "wb") as classifier_f:
         pickle.dump(SGD_classifier,classifier_f)
         classifier_f.close()

      '''
      Apart from training the forest classifier, both .dot and .png files are created with visual
      represntation of the trees
      '''
      print('------------------------------------------------------------------------\n','Random Forest:')
      start_clf_time = time.time()
      rnd_forest = RandomForestClassifier(n_jobs=-1, n_estimators=25, warm_start= True,max_features=7)
      RandomForest_Classifier = SklearnClassifier(rnd_forest)
      RandomForest_Classifier.train(self.training_set)

      if with_trees:
         # Export trees
         i_tree = 0
         for tree_in_forest in rnd_forest.estimators_:
            tree_dot_str = getcwd() +'/trees/tree_' + str(i_tree) + '.dot'
            with open(tree_dot_str, 'w') as tree_dot_file:
               tree_dot_file = tree.export_graphviz(tree_in_forest, out_file=tree_dot_file)

            (graph,) = pydot.graph_from_dot_file(tree_dot_str)
            graph.write_png(tree_dot_str.replace('.dot','.png'))

            i_tree = i_tree + 1

      output = Kappa(RandomForest_Classifier, self.testing_set).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)


      with open(getcwd() + "\\classifiers\\words_as_features\\RandomForest_Classifier.pickle", "wb") as classifier_f:
         pickle.dump(SGD_classifier, classifier_f)
         classifier_f.close()

      print('------------------------------------------------------------------------\n','Adaptive Boosting:')
      start_clf_time = time.time()
      AdaBoost_Classifier = SklearnClassifier(AdaBoostClassifier())
      AdaBoost_Classifier.train(np.array(self.training_set))

      output = Kappa(AdaBoost_Classifier, self.testing_set).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      with open(getcwd() + "\\classifiers\\words_as_features\\AdaBoost_Classifier.pickle", "wb") as classifier_f:
         pickle.dump(SGD_classifier, classifier_f)
         classifier_f.close()

      print('------------------------------------------------------------------------\n','Voted Classifier:')
      start_clf_time = time.time()
      voted_classifier = VoteClassifier(Naivebayes_classifier,
                                        # SVR_classifier,
                                        MLP_Classifier,
                                        RandomForest_Classifier,
                                        # QDA_Classifier,
                                        AdaBoost_Classifier,
                                        SVC_lin_classifier,
                                        # SVC_poly_classifier,
                                        SVC_sig_classifier,
                                        SVC_classifier,
                                        SGD_classifier,
                                        MNB_classifier,
                                        BernoulliNB_classifier,
                                        LogisticRegression_classifier)

      with open(getcwd() + "\\classifiers\\words_as_features\\voted_classifier.pickle", "wb") as classifier_f:
         pickle.dump(SGD_classifier, classifier_f)
         classifier_f.close()

      output = Kappa(voted_classifier, self.testing_set).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      print('------------------------------------------------------------------------')

      self.output_log['Train_News'] =  self.sizes_df.loc['Training']['News']
      self.output_log['Train_Spam'] =  self.sizes_df.loc['Training']['Not-News']
      self.output_log['Test_News']  =  self.sizes_df.loc['Testing']['News']
      self.output_log['Test_Spam']  =  self.sizes_df.loc['Testing']['Not-News']
      self.output_log['feature_cnt'] = self.feature_cnt

      if ngrams:
         self.output_log['type'] = 'bag_of_ngrams_' +str(n_min) + '_'+str(n_max)
      else:
         self.output_log['type'] = 'bag_of_words'

      # Reorder ouput log
      self.output_log = self.output_log[[
         # ID
         'time_stamp','Name','Kernel','feature_cnt','type',
         # Sizes
         'Train_News','Train_Spam','Test_News','Test_Spam',
         'True_News','True_Spam','False_News','False_Spam',

         # Measures
         'Accuracy','Kappa','rauc','duration',
         'News_TPR','News_FPR','News_Prec','News_Recall','News_F1',
         'Spam_TPR','Spam_FPR','Spam_Prec','Spam_Recall','Spam_F1',
      ]]

      # Saving results to file
      df = pd.DataFrame()
      if os.path.isfile(getcwd() + "\\classifiers\\words_as_features\\weighted_confs.csv"):
         retry = 5
         while retry > 0:
            try:
               df = pd.DataFrame().from_csv(getcwd() + "\\classifiers\\words_as_features\\weighted_confs.csv", sep=";")
            except Exception as e:
               retry -= 1
               time.sleep(60)
               print('Error reading file.', retry, 'attempts remainig ...')
               continue
            break

         df = self.output_log.append(df,ignore_index=True)
      else:
         df = self.output_log

      retry = 5
      while retry > 0:
         try:
            df.to_csv(getcwd() + "\\classifiers\\words_as_features\\weighted_confs.csv", sep=";")
            print('saved to', getcwd() + "\\classifiers\\words_as_features\\weighted_confs.csv")
         except Exception as e:
            retry -= 1
            time.sleep(60)
            print('Error writing to file.',retry,'attempts remainig ...')
            continue
         break

   def load_classifier(self):
      with open(getcwd() + "\\classifiers\\words_as_features\\BernoulliNB_classifier.pickle",'rb') as fid:
         BernoulliNB_classifier = pickle.load(fid)
         fid.close()

      # with open(getcwd() + "\\classifiers\\words_as_features\\LinearSVC_classifier.pickle",'rb') as fid:
      #    LinearSVC_classifier = pickle.load(fid)
      #    fid.close()

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

      # -------------------------------- SVM ------------------------------------------------

      with open(getcwd() + "\\classifiers\\words_as_features\\SVC_sigmoid.pickle", 'rb') as fid:
         SVC_sig_classifier = pickle.load(fid)
         fid.close()

      with open(getcwd() + "\\classifiers\\words_as_features\\SVC_poly.pickle", 'rb') as fid:
         SVC_poly_classifier = pickle.load(fid)
         fid.close()

      with open(getcwd() + "\\classifiers\\words_as_features\\SVC_lin.pickle", 'rb') as fid:
         SVC_lin_classifier = pickle.load(fid)
         fid.close()

      with open(getcwd() + "\\classifiers\\words_as_features\\SVC_rbf.pickle", 'rb') as fid:
         SVC_classifier = pickle.load(fid)
         fid.close()

      # -------------------------------------------------------------------------------------


      with open(getcwd() + "\\classifiers\\words_as_features\\words.pickle", 'rb') as fid:
         self.word_features = pickle.load(fid)
         fid.close()

      with open(getcwd() + "\\classifiers\\words_as_features\\voted_classifier.pickle", 'rb') as fid:
         self.voter = pickle.load(fid)
         fid.close()


# def classification_simulation(pth,feature_cnt):
#    #            0    1    2    3    4    5    6    7    8    9
#    num_suff = ['th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th']
#
#    print('Dataset', pth)
#    print('with',feature_cnt,'features: \n')
#    i=1
#    while i <=10:
#       start_time = time.time()
#       print(str(i)+num_suff[(i%9)],'Itterance')
#
#       if i==1:
#          # word_classifier = words_as_features.WordsClassifier('train', pth=pth, from_server=True,num_features=3000)
#          word_classifier = WordsClassifier(n_min=2, n_max=5, ngrams=True, remove_stopwords=True,
#                                                              load_train='train', pth=pth, from_server=True,
#                                                              num_features=feature_cnt, with_trees=False)
#       else:
#          word_classifier = WordsClassifier(n_min=2, n_max=5, ngrams=True, remove_stopwords=True,
#                                                              load_train='train', pth=pth, from_server=False,
#                                                              num_features=feature_cnt, with_trees=False)
#
#       dur = time.strftime('%H:%M:%S',time.gmtime(time.time() - start_time))
#       print('Finished',str(i)+" rounds. Round duration:", str(dur),'\n')
#       i += 1




