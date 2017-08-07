from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.classify import ClassifierI
from nltk import NaiveBayesClassifier,classify
from nltk import pos_tag

from sklearn.svm import SVC, LinearSVC, SVR

from sklearn.metrics import cohen_kappa_score,confusion_matrix
import numpy as np
import scipy.stats as stats
import os




from string import punctuation
from statistics import mode
import random
import pickle
from re import MULTILINE,sub
from os import getcwd,get_exec_path
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt

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
         'Accuracy': self.accuracy
      })
      self.output.name = self.clf_name

      print()
      print(self.prec_recall[['True','False','TPR','FPR','Prec','Recall','F1']])
      print('Accuracy:', self.accuracy,'%')
      print('Kappa:   ', self.kappa_value,'%')

class WordsClassifier():
   '''
   Class must be initialized elsewhere. The class get a path parameter and outputs
   classification statistics for given dataset.
   @:param(load_train)  - indicates whether new classifier have to be trained 
                          or existing ones loaded from pickles
   @:param(pth)         - loaction of dataset
   @:param(from_server) - If true, all tweets will be fetched from server. Else, tweets will be loaded
                          from last downloaded tweets file
   '''
   def __init__(self,gamma,c_par,load_train='load',pth = '',from_server=True,num_features=5000):
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


      self.training_set = None
      self.testining_set = None

      if load_train == 'load':
         self.load_classifier()
      elif load_train == 'train':
         self.train(time_stamp=self.time_stamp,gamma=gamma,c_par=c_par,num_features=num_features,pth=pth,fetch_from_server=from_server)

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

   def train(self,time_stamp,gamma,c_par,pth,num_features,with_print=True,fetch_from_server=True):
      print(pth)

      if fetch_from_server:
         self.fetch_tweets(with_print=with_print,pth=pth)
      else:
         self.load_tweets_from_file()

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
      self.word_features=sorted(self.all_words.items(), key=lambda x:x[1],reverse=True)[10:(num_features+10)]
      self.word_features = [w[0] for w in self.word_features]

      random.shuffle(self.documents)
      featuresets = [(self.find_features(rev,self.word_features), category) for (rev, category) in self.documents]

      with open(getcwd()+"\\classifiers\\words_as_features\\Words.pickle", "wb") as fid:
         pickle.dump(self.word_features, fid)
         fid.close()

      # Sizes
      training_set_size = int(len(featuresets)*0.7)
      self.training_set = featuresets[:training_set_size]
      self.testing_set = featuresets[training_set_size:]
      train_news = sum([1 for obs in self.training_set if obs[1]=='news'])
      train_spam = training_set_size - train_news
      test_news = sum([1 for obs in self.testing_set if obs[1] == 'news'])
      test_spam = len(self.testing_set) - test_news


      if with_print:
         sizes_df = pd.DataFrame(columns=['News','Not-News','Total'])
         sizes_df.loc['Training'] = pd.Series({'News':train_news,'Not-News':train_spam, 'Total':len(self.training_set)})
         sizes_df.loc['Testing'] = pd.Series({'News': test_news, 'Not-News': test_spam, 'Total': len(self.testing_set)})
         sizes_df.loc['Total'] = pd.Series({'News': train_news+test_news,
                                            'Not-News': train_spam+test_spam,
                                            'Total': len(self.training_set)+len(self.testing_set)})
         print(sizes_df.astype(int))

      '''
      ================================================================================================================================================
      ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS 
      ================================================================================================================================================
      '''

      print('------------------------------------------------------------------------\n','C-Support Vector Machine:')
      print('======================\n','Linear Kernel')
      kern,gamm = 'linear',gamma
      SVC_lin_classifier = SklearnClassifier(SVC(kernel=kern,gamma=gamm,C=c_par))
      SVC_lin_classifier.train(self.training_set)

      temp_kappa = Kappa(SVC_lin_classifier, self.testing_set).output
      temp_kappa['Kernel'] = kern
      temp_kappa['gamma'] = gamm
      temp_kappa['C'] = c_par
      self.output_log = self.output_log.append(temp_kappa)
      with open(getcwd()+"\\classifiers\\words_as_features\\SVC_lin.pickle", "wb") as classifier_f:
         pickle.dump(SVC_lin_classifier,classifier_f)
         classifier_f.close()

      print('======================\n', 'Polynomial Kernel')
      kern,gamm = 'poly',gamma
      SVC_poly_classifier = SklearnClassifier(SVC(kernel=kern,gamma=gamm,C=c_par))
      SVC_poly_classifier.train(self.training_set)

      temp_kappa = Kappa(SVC_poly_classifier , self.testing_set).output
      temp_kappa['Kernel'] = kern
      temp_kappa['gamma'] = gamm
      temp_kappa['C'] = c_par
      self.output_log = self.output_log.append(temp_kappa)
      with open(getcwd() + "\\classifiers\\words_as_features\\SVC_poly.pickle", "wb") as classifier_f:
         pickle.dump(SVC_poly_classifier , classifier_f)
         classifier_f.close()

      # Also default kernel
      print('======================\n', 'Radial Basis Function Kernel')
      kern, gamm = 'rbf', gamma
      SVC_classifier = SklearnClassifier(SVC(kernel=kern ,gamma=gamm,C=c_par))
      SVC_classifier.train(self.training_set)

      temp_kappa = Kappa(SVC_classifier, self.testing_set).output
      temp_kappa['Kernel'] = kern
      temp_kappa['gamma'] = gamm
      temp_kappa['C'] = c_par
      self.output_log = self.output_log.append(temp_kappa)
      with open(getcwd() + "\\classifiers\\words_as_features\\SVC_rbf.pickle", "wb") as classifier_f:
         pickle.dump(SVC_classifier, classifier_f)
         classifier_f.close()

      print('======================\n', 'Sigmoid Kernel')
      kern, gamm = 'sigmoid', gamma
      SVC_sig_classifier = SklearnClassifier(SVC(kernel=kern ,gamma=gamm,C=c_par))
      SVC_sig_classifier.train(self.training_set)
      temp_kappa = Kappa(SVC_sig_classifier, self.testing_set).output
      temp_kappa = Kappa(SVC_classifier, self.testing_set).output
      temp_kappa['Kernel'] = kern
      temp_kappa['gamma'] = gamm
      temp_kappa['C'] = c_par

      self.output_log = self.output_log.append(temp_kappa)

      with open(getcwd() + "\\classifiers\\words_as_features\\SVC_sigmoid.pickle", "wb") as classifier_f:
         pickle.dump(SVC_sig_classifier, classifier_f)
         classifier_f.close()

      '''
      ================================================================================================================================================
      '''

      print('------------------------------------------------------------------------')
      self.output_log['time_stamp']=time_stamp
      self.output_log['Train_News'] = sizes_df.loc['Training']['News']
      self.output_log['Train_Spam'] = sizes_df.loc['Training']['Not-News']
      self.output_log['Test_News']  = sizes_df.loc['Testing']['News']
      self.output_log['Test_Spam']  = sizes_df.loc['Testing']['Not-News']

      # Reorder ouput log
      self.output_log = self.output_log[[
         # ID
         'time_stamp','Name','Kernel','gamma','C',
         # Sizes
         'Train_News','Train_Spam','Test_News','Test_Spam',
         'True_News','True_Spam','False_News','False_Spam',

         # Measures
         'Accuracy','Kappa',
         'News_TPR','News_FPR','News_Prec','News_Recall','News_F1',
         'Spam_TPR','Spam_FPR','Spam_Prec','Spam_Recall','Spam_F1',
      ]]

      # Saving results to file

      if os.path.isfile(getcwd() + "\\classifiers\\words_as_features\\kernels_weighted_confs.csv"):
         df = pd.DataFrame().from_csv(getcwd() + "\\classifiers\\words_as_features\\kernels_weighted_confs.csv",sep=";")
         df = self.output_log.append(df,ignore_index=True)
      else:
         df = self.output_log

      df.to_csv(getcwd() + "\\classifiers\\words_as_features\\kernels_weighted_confs.csv",sep=";")


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

