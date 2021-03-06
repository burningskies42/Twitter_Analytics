from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.classify import ClassifierI
from nltk import NaiveBayesClassifier,classify
from nltk import pos_tag

from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
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

import feature_tk.features as fe



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

   def predict(self, features):
      votes = []
      for c in self._classifiers:
         v = c.predict(features)
         votes.append(v)

      votes_t = np.array(votes)
      votes_t = votes_t.transpose()

      a = np.array([mode(v) for v in votes_t])

      return a

   def score(self, Xs,y_test):

      y_predicted = [1 if y==1 else 0 for y in self.predict(Xs)]
      score= sum([1 if y[0]==y[1] else 0 for y in list(zip(y_predicted,y_test))])/len(y_test)
      return(score)

class Kappa():
   '''
   creates a true/false - negative/positive matrix from calculating a Kappa value
   @:param(classifier) - classifier which was trained using a learning algorithm
   '''
   def __init__(self,classifier,X_test,y_test):
      self.kappa_matrix = {'news':{'true':0,'false':0},
                           'spam':{'true':0,'false':0}}
      self.clf = classifier

      # Get classifier name
      try:
         self.clf_name = self.clf._clf.__class__.__name__
      except Exception as e:
         self.clf_name = self.clf.__class__.__name__

      self.clf_name = self.clf_name.replace('Classifier','')

      y_test = [1 if y==1 else 0 for y in y_test]


      Xs = [X for X in X_test]

      y_predictions = np.array(self.clf.predict(Xs))
      y_predictions = [1 if line == 1 else 0 for line in y_predictions]

      # print(y_predictions)
      # print(y_test)
      # quit()

      self.rauc_score = round(roc_auc_score(y_test,y_predictions) * 100, 2)

      # Count False Pos/Neg and True Pos/Neg
      # Where Pos = News and Neg = Spam

      # for fs in testing_set:
      for x,y in zip(Xs,y_test):

      # true classification is 'news'
         alg_class = self.clf.predict(x.reshape(1,-1))[0]
         alg_class = 1 if alg_class==1 else 0

         if y == 1:
            # algo classified correctly (a)
            if alg_class == 1:
               self.kappa_matrix['news']['true'] += 1

            # algo misclassified 'news' as 'spam' (b)
            else:
               self.kappa_matrix['spam']['false'] += 1

         # true classification is 'spam'
         elif y == 0:
            # # algo classified correctly (d)
            if alg_class == 0:
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
      self.accuracy = round(classifier.score(Xs,y_test) * 100, 2)

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


   def str_to_featureset(self, str):
      str = self.clear_urls(str)
      str = self.build_word_list(str)
      featureset = self.find_features(str, self.word_features)

      return featureset

   def fetch_tweets(self,pth,with_print):

      # List of tuples: each tuple contains a list of words(tokenized sentence) + category('pos' or 'neg')
      label_df = pd.DataFrame.from_csv(pth,sep=';')
      label_df.index = label_df.index.map(str)
      label_df['label'] = label_df['label'].apply(lambda x : self.txt_lbl(x) )
      ids = label_df.index.values

      if with_print: print('Dataset size is',len(ids),'tweets')
      if with_print: print('------------------------------------')
      df = fetch_tweets_by_ids(ids)

      df = label_df.join(df,how='right')
      df['id_str'] = df.index

      df.dropna(inplace=True,how='all')
      df.to_csv(getcwd()+'\\classifiers\\words_as_features\\desc_latest_dataset.csv',sep=';')

      self.documents = [(row.drop(['label']).to_dict(),row['label']) for ind,row in df.iterrows()]

      # for a, b in(self.documents):
      #    print(a,b)
      # quit()

      # random.shuffle(self.documents)

      with open(getcwd()+"\\classifiers\\words_as_features\\desc_Documents.pickle", "wb") as fid:
         pickle.dump(self.documents,fid)
         fid.close()
      # self.class_ratio = int(len(df[df['label']=='spam'])/len(df[df['label']=='news']))

   def load_tweets_from_file(self):
      with open(getcwd()+"\\classifiers\\words_as_features\\desc_Documents.pickle", "rb") as fid:
         self.documents = pickle.load(fid)
         fid.close()

   def build_features(self):
      self.load_tweets_from_file()

      df_jsons = pd.DataFrame()

      for jsn, lbl in self.documents:
         se = pd.Series(jsn)
         se['label'] = lbl
         se.name = str(se['id_str'])
         df_jsons = df_jsons.append(se)

      df_features = fe.tweets_to_featureset(df_jsons,with_timing=True,with_sentiment=True)

      df_features.drop(labels=['words','words_no_url'],axis=1,inplace = True)

      # desc = df_features.describe(include = 'all')
      # desc.to_csv('describe_features.csv',sep=';')
      # quit()

      self.featuresets =[(line.drop(labels=['label']).to_dict(),line['label']) for key,line in df_features.iterrows()]


      # for a,b in self.featuresets:
      #    print(b,a)
      # quit()


   def train_test_split(self,with_print):
      # Sizes
      training_set_size = int(len(self.featuresets) * 0.7)
      # self.training_set = self.featuresets[:training_set_size]
      # self.testing_set = self.featuresets[training_set_size:]
      random.shuffle(self.featuresets)

      self.X, self.y = zip(*self.featuresets)

      self.X_train = np.array(pd.DataFrame.from_records(self.X[:training_set_size]))
      self.X_test = np.array(pd.DataFrame.from_records(self.X[training_set_size:]))

      self.X_prep = preprocessing.scale(pd.DataFrame.from_records(self.X))
      self.X_prep_train = np.array(pd.DataFrame.from_records(self.X_prep[:training_set_size]))
      self.X_prep_test = np.array(pd.DataFrame.from_records(self.X_prep[training_set_size:]))

      self.y_train = np.array([(i == 'spam') + 1 for i in self.y[:training_set_size]])
      self.y_test = np.array([(i == 'spam') + 1 for i in self.y[training_set_size:]])



      self.sizes_df = pd.DataFrame()
      se = pd.Series({'News': 0, 'Not-News': 0})
      se.name='Training'
      self.sizes_df = self.sizes_df.append(se)
      se.name='Testing'
      self.sizes_df = self.sizes_df.append(se)


      self.sizes_df.loc['Training']['News']     = sum([1 for obs in self.y_train if obs == 1])
      self.sizes_df.loc['Training']['Not-News'] = training_set_size - self.sizes_df.loc['Training']['News']
      self.sizes_df.loc['Testing']['News']      = sum([1 for obs in self.y_test if obs == 1])
      self.sizes_df.loc['Testing']['Not-News']  = len(self.X_test) - self.sizes_df.loc['Testing']['News']

      print(self.sizes_df)

      print('------------------------------------------------------------------------\n', 'Linear Regression:')
      # This split datasets are only used in linear regression

      # self.X_train, self.y_train = zip(*self.training_set)
      # self.X_train = np.array(pd.DataFrame.from_records(self.X_train))
      #
      # # Covert labels to numbers: 1=News, 2=NotNews
      # self.y_train = np.array([(i == 'spam') + 1 for i in self.y_train])
      #
      # self.X_test, self.y_test = zip(*self.testing_set)
      # self.X_test = np.array(pd.DataFrame.from_records(self.X_test))
      #
      # self.y_test = np.array([(i == 'spam') + 1 for i in self.y_test])

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

   def train(self,with_trees,with_print):
      # if fetch_from_server:
      #    self.fetch_tweets(with_print=with_print,pth=pth,remove_stopwords=remove_stopwords,ngrams=ngrams,n_min=n_min,n_max=n_max)
      # else:

      # self.train_test_split(with_print)

      # Logistic Regression
      print('------------------------------------------------------------------------\n','Logistic Regression:')
      start_clf_time = time.time()
      LogisticRegression_classifier = LogisticRegression(fit_intercept=True)

      LogisticRegression_classifier.fit(X=self.X_train,y=self.y_train)
      output = Kappa(LogisticRegression_classifier,X_test= self.X_test,y_test=self.y_test).output

      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      with open(getcwd()+"\\classifiers\\words_as_features\\LogisticRegression.pickle", "wb") as classifier_f:
         pickle.dump(LogisticRegression_classifier,classifier_f)
         classifier_f.close()

      print('------------------------------------------------------------------------\n','Naive Bayes:')
      start_clf_time = time.time()
      Naivebayes_classifier = GaussianNB()

      Naivebayes_classifier.fit(X=self.X_train,y=self.y_train)
      output = Kappa(Naivebayes_classifier, X_test=self.X_test, y_test=self.y_test).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      # Naivebayes_classifier.show_most_informative_features(15)

      with open(getcwd()+"\\classifiers\\words_as_features\\Naivebayes_classifier.pickle", "wb") as classifier_f:
         pickle.dump(Naivebayes_classifier,classifier_f)
         classifier_f.close()

      print('------------------------------------------------------------------------\n','Multinomial Naive Bayes:')
      start_clf_time = time.time()
      MNB_classifier =  MultinomialNB()

      MNB_classifier.fit(X=self.X_train,y=self.y_train)
      output = Kappa(MNB_classifier, X_test=self.X_test, y_test=self.y_test).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      with open(getcwd()+"\\classifiers\\words_as_features\\MNB_classifier.pickle", "wb") as classifier_f:
         pickle.dump(MNB_classifier,classifier_f)
         classifier_f.close()

      print('------------------------------------------------------------------------\n','Bernoulli Naive Bayes:')
      start_clf_time = time.time()
      BernoulliNB_classifier = BernoulliNB()
      BernoulliNB_classifier.fit(X=self.X_train,y=self.y_train)

      output = Kappa(BernoulliNB_classifier, X_test=self.X_test, y_test=self.y_test).output
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
      SVC_lin_classifier = SVC(kernel='linear')
      SVC_lin_classifier.fit(X=self.X_prep_train,y=self.y_train)

      output = Kappa(SVC_lin_classifier, X_test=self.X_prep_test, y_test=self.y_test).output
      output['Kernel'] = 'linear'
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      with open(getcwd()+"\\classifiers\\words_as_features\\SVC_lin.pickle", "wb") as classifier_f:
         pickle.dump(SVC_lin_classifier,classifier_f)
         classifier_f.close()

      print('======================\n', 'Polynomial Kernel')
      start_clf_time = time.time()
      SVC_poly_classifier = SVC(kernel='poly',C=1,gamma=1)
      SVC_poly_classifier.fit(X=self.X_prep_train,y=self.y_train)

      output = Kappa(SVC_poly_classifier, X_test = self.X_prep_test, y_test=self.y_test).output
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
      SVC_classifier = SVC(kernel='rbf',gamma=0.1,C=1.38)
      SVC_classifier.fit(X=self.X_prep_train,y=self.y_train)

      output = Kappa(SVC_classifier, X_test=self.X_prep_test, y_test=self.y_test).output
      output['Kernel'] = 'rbf'
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      with open(getcwd() + "\\classifiers\\words_as_features\\SVC_rbf.pickle", "wb") as classifier_f:
         pickle.dump(SVC_classifier, classifier_f)
         classifier_f.close()

      print('======================\n', 'Sigmoid Kernel')
      start_clf_time = time.time()
      SVC_sig_classifier = SVC(kernel='sigmoid',gamma=10)
      SVC_sig_classifier.fit(X=self.X_prep_train,y=self.y_train)

      output = Kappa(SVC_sig_classifier, X_test=self.X_prep_test, y_test=self.y_test).output
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
      SGD_classifier = SGDClassifier()
      SGD_classifier.fit(X=self.X_train,y=self.y_train)

      output = Kappa(SGD_classifier, X_test=self.X_test, y_test=self.y_test).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      with open(getcwd()+"\\classifiers\\words_as_features\\SGD_classifier.pickle", "wb") as classifier_f:
         pickle.dump(SGD_classifier,classifier_f)
         classifier_f.close()

      print('------------------------------------------------------------------------\n','Multi-layer Perceptron:')
      start_clf_time = time.time()
      MLP_Classifier = MLPClassifier(alpha=1)
      MLP_Classifier.fit(X=self.X_train,y=self.y_train)

      output = Kappa(MLP_Classifier, X_test=self.X_test, y_test=self.y_test).output
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
      RandomForest_Classifier = rnd_forest
      RandomForest_Classifier.fit(X=self.X_train,y=self.y_train)

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

      output = Kappa(RandomForest_Classifier, X_test=self.X_test, y_test=self.y_test).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)


      with open(getcwd() + "\\classifiers\\words_as_features\\RandomForest_Classifier.pickle", "wb") as classifier_f:
         pickle.dump(SGD_classifier, classifier_f)
         classifier_f.close()

      print('------------------------------------------------------------------------\n','Adaptive Boosting:')
      start_clf_time = time.time()
      AdaBoost_Classifier = AdaBoostClassifier()
      AdaBoost_Classifier.fit(X=self.X_train,y=self.y_train)

      output = Kappa(AdaBoost_Classifier, X_test=self.X_test, y_test=self.y_test).output
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

      output = Kappa(voted_classifier, X_test=self.X_test, y_test=self.y_test).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      print('------------------------------------------------------------------------')

      self.output_log['Train_News'] =  self.sizes_df.loc['Training']['News']
      self.output_log['Train_Spam'] =  self.sizes_df.loc['Training']['Not-News']
      self.output_log['Test_News']  =  self.sizes_df.loc['Testing']['News']
      self.output_log['Test_Spam']  =  self.sizes_df.loc['Testing']['Not-News']
      self.output_log['feature_cnt'] = None

      self.output_log['type'] = 'descriptive_features'

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
      if os.path.isfile(getcwd() + "\\classifiers\\words_as_features\\desc_weighted_confs.csv"):
         retry = 5
         while retry > 0:
            try:
               df = pd.DataFrame().from_csv(getcwd() + "\\classifiers\\words_as_features\\desc_weighted_confs.csv", sep=";")
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
            df.to_csv(getcwd() + "\\classifiers\\words_as_features\\desc_weighted_confs.csv", sep=";")
            print('saved to', getcwd() + "\\classifiers\\words_as_features\\desc_weighted_confs.csv")
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




