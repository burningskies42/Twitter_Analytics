import time
import easygui

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
from sklearn.metrics import roc_auc_score,roc_curve,auc,accuracy_score
import matplotlib.pyplot as plt
from sklearn import tree

import pydot
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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

from tweet_tk.fetch_tweets import fetch_tweets_by_ids
from labeled_featureset_builder import open_and_join

class VoteClassifier(ClassifierI):
   def __init__(self, *classifiers):
      self._classifiers = classifiers

   def predict(self, features):
      votes = []
      for c in self._classifiers:
         v=[]
         for each in features:
            v.append(c.predict(each[0]))
            print(v)

         votes.append(mode(v))

      print(votes)
      quit()
      return mode(votes)

   def confidence(self, features):
      votes = []
      for c in self._classifiers:
         v = c.predict(features)
         votes.append(v)

      choice_votes = votes.count(mode(votes))
      conf = choice_votes / len(votes)
      return conf

class Kappa():
   '''
   creates a true/false - negative/positive matrix from calculating a Kappa value
   @:param(classifier) - classifier which was trained using a learning algorithm
   '''
   def __init__(self,classifier,testing_set,prep=False):
      self.kappa_matrix = {'news':{'true':0,'false':0},
                           'spam':{'true':0,'false':0}}
      self.clf = classifier

      # Get classifier name
      try:
         self.clf_name = self.clf._clf.__class__.__name__
      except Exception as e:
         self.clf_name = self.clf.__class__.__name__

      self.clf_name = self.clf_name.replace('Classifier','')

      X_test = testing_set.drop(['label'], axis=1)
      X_test = np.array(pd.DataFrame.from_records(X_test))

      if prep:
         X_test = preprocessing.scale(X_test)

      y_test = testing_set['label'].tolist()
      y_test = np.array([0 if y == 2 else 1 for y in y_test])

      y_predictions = self.clf.predict(X_test)


      self.rauc_score = round(roc_auc_score(y_test,y_predictions) * 100, 2)

      # Count False Pos/Neg and True Pos/Neg
      # Where Pos = News and Neg = Spam
      for actual,prediction in zip(y_test,y_predictions):

         # true classification is 'news'
         alg_class = prediction

         if actual == 1:
            # algo classified correctly (a)
            if alg_class == 1:
               self.kappa_matrix['news']['true'] += 1

            # algo misclassified 'news' as 'spam' (b)
            else:
               self.kappa_matrix['spam']['false'] += 1

         # true classification is 'spam'
         elif actual == 0:
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
      self.accuracy = round(accuracy_score(y_true=y_test,y_pred=y_predictions) * 100, 2)

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
   Class must be initialized elsewhere. The class get a path parameter and outputs
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

      if load_train == 'load':
         self.load_classifier()
      elif load_train == 'train':
         self.train(self.time_stamp,num_features=num_features,pth=pth,fetch_from_server=from_server,with_trees=self.with_trees)


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
      # df = fetch_tweets_by_ids(ids)[['text']]

      df = open_and_join(pth, True, with_sentiment=False, with_timing=False)
      df.drop(['words','words_no_url'],inplace=True,axis=1)

      df.dropna(inplace=True)
      df.to_csv(getcwd()+'\\classifiers\\words_as_features\\desc_latest_dataset.csv',sep=';')

      # self.documents = [(row['text'],row['label']) for ind,row in df.iterrows()]
      # print(self.documents)
      self.documents = df

      # random.shuffle(self.documents)

      # with open(getcwd()+"\\classifiers\\words_as_features\\Documents.pickle", "wb") as fid:
      #    pickle.dump(self.documents,fid)
      #    fid.close()
      # self.class_ratio = int(len(df[df['label']=='spam'])/len(df[df['label']=='news']))

   def load_tweets_from_file(self):
      with open(getcwd()+"\\classifiers\\words_as_features\\desc_latest_dataset.pickle", "rb") as fid:
         self.documents = pickle.load(fid)
         fid.close()

   def train(self,time_stamp,pth,num_features,with_trees,with_print=True,fetch_from_server=True):
      print(pth)

      if fetch_from_server:
         self.fetch_tweets(with_print=with_print,pth=pth)
      else:
         self.load_tweets_from_file()

      # for tweet in self.documents:
      #    if tweet[1] == 'news' :
      #       mult = self.class_ratio
      #    else:
      #       mult = 1
      #
      #    for word in tweet[0]:
      #       if word.lower() in self.all_words.keys():
      #          self.all_words[word.lower()] += 1*mult
      #       else:
      #          self.all_words[word.lower()] = 1*mult

      # Get the 5000 most popular words
      # self.word_features=sorted(self.all_words.items(), key=lambda x:x[1],reverse=True)[10:(num_features+10)]
      # self.word_features = [w[0] for w in self.word_features]
      #
      # random.shuffle(self.documents)
      # featuresets = [(self.find_features(rev,self.word_features), category) for (rev, category) in self.documents]
      #
      # with open(getcwd()+"\\classifiers\\words_as_features\\Words.pickle", "wb") as fid:
      #    pickle.dump(self.word_features, fid)
      #    fid.close()
      featuresets = self.documents


      # Sizes
      training_set_size = int(len(featuresets)*0.7)
      self.training_set = featuresets[:training_set_size]
      self.testing_set = featuresets[training_set_size:]


      # train_news = sum([1 for obs in self.training_set if obs['label']==1])
      train_news = self.training_set[self.training_set['label']==1]['label'].count()
      train_spam = training_set_size - train_news

      test_news = self.testing_set[self.testing_set['label']==1]['label'].count()
      test_spam = len(self.testing_set) - test_news


      if with_print:
         sizes_df = pd.DataFrame(columns=['News','Not-News','Total'])
         sizes_df.loc['Training'] = pd.Series({'News':train_news,'Not-News':train_spam, 'Total':len(self.training_set)})
         sizes_df.loc['Testing'] = pd.Series({'News': test_news, 'Not-News': test_spam, 'Total': len(self.testing_set)})
         sizes_df.loc['Total'] = pd.Series({'News': train_news+test_news,
                                            'Not-News': train_spam+test_spam,
                                            'Total': len(self.training_set)+len(self.testing_set)})
         print(sizes_df.astype(int))

      print('------------------------------------------------------------------------\n', 'Logistic Regression:')
      # This split datasets are only used in linear regression

      X_train = self.training_set.drop(['label'], axis=1)
      X_train = np.array(pd.DataFrame.from_records(X_train))
      X_train_prep = preprocessing.scale(X_train)

      X_test = self.testing_set.drop(['label'], axis=1)
      X_test = np.array(pd.DataFrame.from_records(X_test))
      X_test_prep = preprocessing.scale(X_test)

      y_train = self.training_set['label'].tolist()
      y_train = np.array([0 if y==2 else 1 for y in y_train])

      y_test = self.testing_set['label'].tolist()
      y_test = np.array([0 if y==2 else 1 for y in y_test])


      LinearRegression_classifier = LinearRegression()
      LinearRegression_classifier.fit(X_train,y_train)
      R2 = round(LinearRegression_classifier.score(X_test,y_test),2)

      news_ys = []
      spam_ys = []
      for x,y in zip(X_test,y_test):
         y_hat = LinearRegression_classifier.predict(x.reshape(1, -1))[0]
         if y == 1:
            news_ys.append(y_hat)
         else:
            spam_ys.append(y_hat)

      if with_print:
         gen_stats_df = pd.DataFrame(columns=['News','Not-News'])
         gen_stats_df.loc['mean'] = pd.Series({'News':round(np.mean(news_ys),2),'Not-News':round(np.mean(spam_ys),2)})
         gen_stats_df.loc['median'] = pd.Series({'News': round(np.median(news_ys), 2), 'Not-News': round(np.median(spam_ys), 2)})
         print(gen_stats_df)
         print('Coeffiecient of Determination (R^2):', R2)

      # Logistic Regression
      print('------------------------------------------------------------------------\n','Logistic Regression:')
      start_clf_time = time.time()
      LogisticRegression_classifier = LogisticRegression(fit_intercept=True)

      LogisticRegression_classifier.fit(X=X_train, y = y_train)

      output = Kappa(LogisticRegression_classifier, self.testing_set).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      # with open(getcwd()+"\\classifiers\\words_as_features\\LogisticRegression.pickle", "wb") as classifier_f:
      #    pickle.dump(LogisticRegression_classifier,classifier_f)
      #    classifier_f.close()

      print('------------------------------------------------------------------------\n','Naive Bayes:')
      start_clf_time = time.time()
      # Naivebayes_classifier = NaiveBayesClassifier.fit(X=X_train, y = y_train)
      #
      # output = Kappa(Naivebayes_classifier, self.testing_set).output
      # output['duration'] =  round(time.time() - start_clf_time,3)
      # output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      # self.output_log = self.output_log.append(output)
      #
      # Naivebayes_classifier.show_most_informative_features(15)
      #
      # with open(getcwd()+"\\classifiers\\words_as_features\\Naivebayes_classifier.pickle", "wb") as classifier_f:
      #    pickle.dump(Naivebayes_classifier,classifier_f)
      #    classifier_f.close()

      print('------------------------------------------------------------------------\n','Multinomial Naive Bayes:')
      start_clf_time = time.time()
      MNB_classifier = MultinomialNB()
      MNB_classifier.fit(X=X_train, y = y_train)

      output = Kappa(MNB_classifier, self.testing_set).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      # with open(getcwd()+"\\classifiers\\words_as_features\\MNB_classifier.pickle", "wb") as classifier_f:
      #    pickle.dump(MNB_classifier,classifier_f)
      #    classifier_f.close()

      print('------------------------------------------------------------------------\n','Bernoulli Naive Bayes:')
      start_clf_time = time.time()
      BernoulliNB_classifier = BernoulliNB()
      BernoulliNB_classifier.fit(X=X_train, y = y_train)

      output = Kappa(BernoulliNB_classifier, self.testing_set).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      # with open(getcwd()+"\\classifiers\\words_as_features\\BernoulliNB_classifier.pickle", "wb") as classifier_f:
      #    pickle.dump(BernoulliNB_classifier,classifier_f)
      #    classifier_f.close()


      '''
      ================================================================================================================================================
      ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS ~~~ SVM KERNELS 
      ================================================================================================================================================
      '''

      print('------------------------------------------------------------------------\n','C-Support Vector Machine:')
      print('======================\n','Linear Kernel')
      start_clf_time = time.time()
      SVC_lin_classifier = SVC(kernel='linear')
      SVC_lin_classifier.fit(X=X_train_prep, y = y_train)

      output = Kappa(SVC_lin_classifier, self.testing_set,prep=True).output
      output['Kernel'] = 'linear'
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      # with open(getcwd()+"\\classifiers\\words_as_features\\SVC_lin.pickle", "wb") as classifier_f:
      #    pickle.dump(SVC_lin_classifier,classifier_f)
      #    classifier_f.close()

      print('======================\n', 'Polynomial Kernel')
      start_clf_time = time.time()
      SVC_poly_classifier = SVC(kernel='poly',C=1,gamma=1)
      SVC_poly_classifier.fit(X=X_train_prep, y = y_train)

      output = Kappa(SVC_poly_classifier, self.testing_set,prep=True).output
      output['Kernel'] = 'poly'
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      # with open(getcwd() + "\\classifiers\\words_as_features\\SVC_poly.pickle", "wb") as classifier_f:
      #    pickle.dump(SVC_poly_classifier , classifier_f)
      #    classifier_f.close()

      # Also default kernel
      print('======================\n', 'Radial Basis Function Kernel')
      start_clf_time = time.time()
      SVC_classifier = SVC(kernel='rbf',gamma=0.1,C=1.38)
      SVC_classifier.fit(X=X_train_prep, y = y_train)

      output = Kappa(SVC_classifier, self.testing_set,prep=True).output
      output['Kernel'] = 'rbf'
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      # with open(getcwd() + "\\classifiers\\words_as_features\\SVC_rbf.pickle", "wb") as classifier_f:
      #    pickle.dump(SVC_classifier, classifier_f)
      #    classifier_f.close()

      print('======================\n', 'Sigmoid Kernel')
      start_clf_time = time.time()
      SVC_sig_classifier = SVC(kernel='sigmoid',gamma=10)
      SVC_sig_classifier.fit(X=X_train_prep, y = y_train)

      output = Kappa(SVC_sig_classifier, self.testing_set,prep=True).output
      output['Kernel'] = 'sigmoid'
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      # with open(getcwd() + "\\classifiers\\words_as_features\\SVC_sigmoid.pickle", "wb") as classifier_f:
      #    pickle.dump(SVC_sig_classifier, classifier_f)
      #    classifier_f.close()

      '''
      ================================================================================================================================================
      '''

      print('------------------------------------------------------------------------\n','Stochastic Gradient Descent:')
      start_clf_time = time.time()
      SGD_classifier = SGDClassifier()
      SGD_classifier.fit(X=X_train, y = y_train)

      output = Kappa(SGD_classifier, self.testing_set).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      # with open(getcwd()+"\\classifiers\\words_as_features\\SGD_classifier.pickle", "wb") as classifier_f:
      #    pickle.dump(SGD_classifier,classifier_f)
      #    classifier_f.close()

      print('------------------------------------------------------------------------\n','Multi-layer Perceptron:')
      start_clf_time = time.time()
      MLP_Classifier = MLPClassifier(alpha=1)
      MLP_Classifier.fit(X=X_train, y = y_train)

      output = Kappa(MLP_Classifier, self.testing_set).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      # with open(getcwd()+"\\classifiers\\words_as_features\\MLP_Classifier.pickle", "wb") as classifier_f:
      #    pickle.dump(SGD_classifier,classifier_f)
      #    classifier_f.close()

      '''
      Apart from training the forest classifier, both .dot and .png files are created with visual
      represntation of the trees
      '''
      print('------------------------------------------------------------------------\n','Random Forest:')
      start_clf_time = time.time()
      rnd_forest = RandomForestClassifier(n_jobs=-1, n_estimators=25, warm_start= True,max_features=7)
      RandomForest_Classifier = rnd_forest
      RandomForest_Classifier.fit(X=X_train, y = y_train)

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


      # with open(getcwd() + "\\classifiers\\words_as_features\\RandomForest_Classifier.pickle", "wb") as classifier_f:
      #    pickle.dump(SGD_classifier, classifier_f)
      #    classifier_f.close()

      print('------------------------------------------------------------------------\n','Adaptive Boosting:')
      start_clf_time = time.time()
      AdaBoost_Classifier = AdaBoostClassifier()
      AdaBoost_Classifier.fit(X=X_train, y = y_train)

      output = Kappa(AdaBoost_Classifier, self.testing_set).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      with open(getcwd() + "\\classifiers\\words_as_features\\AdaBoost_Classifier.pickle", "wb") as classifier_f:
         pickle.dump(SGD_classifier, classifier_f)
         classifier_f.close()

      print('------------------------------------------------------------------------\n','Voted Classifier:')
      start_clf_time = time.time()
      voted_classifier = VoteClassifier(
                                        # Naivebayes_classifier,
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

      output = Kappa(voted_classifier, self.testing_set,prep=True).output
      output['duration'] =  round(time.time() - start_clf_time,3)
      output['time_stamp'] = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
      self.output_log = self.output_log.append(output)

      print('------------------------------------------------------------------------')

      self.output_log['Train_News'] = sizes_df.loc['Training']['News']
      self.output_log['Train_Spam'] = sizes_df.loc['Training']['Not-News']
      self.output_log['Test_News']  = sizes_df.loc['Testing']['News']
      self.output_log['Test_Spam']  = sizes_df.loc['Testing']['Not-News']
      self.output_log['feature_cnt'] = num_features
      self.output_log['type'] = 'descriptive'

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

      if os.path.isfile(getcwd() + "\\classifiers\\words_as_features\\weighted_confs.csv"):
         df = pd.DataFrame().from_csv(getcwd() + "\\classifiers\\words_as_features\\weighted_confs.csv",sep=";")
         df = self.output_log.append(df,ignore_index=True)
      else:
         df = self.output_log

      df.to_csv(getcwd() + "\\classifiers\\words_as_features\\weighted_confs.csv",sep=";")


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


pth = easygui.fileopenbox()
#            0    1    2    3    4    5    6    7    8    9
num_suff = ['th','st','nd','rd','th','th','th','th','th','th']


for feature_cnt in [100,500,1000,3000,5000]:
   i=1
   total_run_time = time.time()
   while i <=10:
      start_time = time.time()
      print(str(i)+num_suff[(i%9)],'Itterance')

      if i==1:
         # word_classifier = words_as_features.WordsClassifier('train', pth=pth, from_server=True,num_features=3000)
         word_classifier = WordsClassifier('train', pth=pth, from_server=True,num_features=feature_cnt,with_trees = False)
      else:
         word_classifier = WordsClassifier('train', pth=pth, from_server=False,num_features=feature_cnt,with_trees = False)

      dur = time.strftime('%H:%M:%S',time.gmtime(time.time() - start_time))
      print('Finished',str(i)+" rounds. Round duration:", str(dur))
      i += 1



print('Total run-time',str(time.strftime('%H:%M:%S',time.gmtime(time.time() - total_run_time))))