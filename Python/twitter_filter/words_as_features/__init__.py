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
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.metrics import cohen_kappa_score,confusion_matrix
import numpy as np
import scipy.stats as stats
import pylab as pl


from string import punctuation
from statistics import mode
import random
import pickle
from re import MULTILINE,sub
from os import getcwd,get_exec_path
import pandas as pd
import time

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
      self.kappa_matrix = {}
      self.kappa_matrix['news'] = {}
      self.kappa_matrix['news']['true'] = 0
      self.kappa_matrix['news']['false'] = 0

      self.kappa_matrix['spam'] = {}
      self.kappa_matrix['spam']['true'] = 0
      self.kappa_matrix['spam']['false'] = 0
      self.clf = classifier

      for fs in testing_set:
         # true classification is 'news'

         # if reg:
         #    alg_class = self.clf.predict(fs[0])
         # else:
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

      for key, val in self.kappa_matrix.items():
         print(key, val)

      # https://en.wikipedia.org/wiki/Cohen%27s_kappa

      a = self.kappa_matrix['news']['true']
      b = self.kappa_matrix['spam']['false']
      c = self.kappa_matrix['news']['false']
      d = self.kappa_matrix['spam']['true']
      abcd = a+b+c+d
      p_zero = (a+d)/abcd
      p_news = ((a + b) / abcd) * ((a + c) / abcd)
      p_spam = ((c + d) / abcd) * ((b + d) / abcd)
      p_e = p_news + p_spam

      self.kappa_value = (p_zero - p_e) / (1 - p_e)
      print("Cohen's Kappa:",self.kappa_value)

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
   def __init__(self,load_train='load',pth = '',from_server=True):
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
      self.accuracies = {}
      self.kappas = {}
      self.class_ratio = 1

      self.training_set = None
      self.testining_set = None

      if load_train == 'load':
         self.load_classifier()
      elif load_train == 'train':
         self.train(pth=pth,fetch_from_server=from_server)

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

      with open(getcwd()+"\\classifiers\\words_as_features\\Documents.pickle", "wb") as fid:
         pickle.dump(self.documents,fid)
         fid.close()
      # self.class_ratio = int(len(df[df['label']=='spam'])/len(df[df['label']=='news']))

   def load_tweets_from_file(self):
      with open(getcwd()+"\\classifiers\\words_as_features\\Documents.pickle", "rb") as fid:
         self.documents = pickle.load(fid)
         fid.close()

   def train(self,pth,with_print=True,fetch_from_server=True):
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
      self.word_features=sorted(self.all_words.items(), key=lambda x:x[1],reverse=True)[:5000]
      self.word_features = [w[0] for w in self.word_features]

      featuresets = [(self.find_features(rev,self.word_features), category) for (rev, category) in self.documents]

      with open(getcwd()+"\\classifiers\\words_as_features\\Words.pickle", "wb") as fid:
         pickle.dump(self.word_features, fid)
         fid.close()

      training_set_size = int(len(featuresets)*0.7)

      self.training_set = featuresets[:training_set_size]
      self.testing_set = featuresets[training_set_size:]

      train_news = sum([1 for obs in self.training_set if obs[1]=='news'])
      train_spam = training_set_size - train_news
      test_news = sum([1 for obs in self.testing_set if obs[1] == 'news'])
      test_spam = len(self.testing_set) - test_news

      self.accuracies['train_news'] = train_news
      self.accuracies['train_spam'] = train_spam
      self.accuracies['test_news'] = test_news
      self.accuracies['test_spam'] = test_spam

      if with_print:
         print()
         print('            News    Not-News   Total')
         print('Training   ', train_news,'   ',train_spam,'     ',len(self.training_set))
         print('Testing    ', test_news,'   ', test_spam,'     ', len(self.testing_set))
         print()

      self.accuracies['training_set_len'] = len(self.training_set)
      self.accuracies['testing_set_len'] = len(self.testing_set)

      # Linear Regression
      print('------------------------------------------------------------------------')
      print('Linear Regression:')
      # LinearRegression_classifier = SklearnClassifier(LinearRegression(),sparse=False)
      # LinearRegression_classifier.train(self.training_set)
      # LinReg_accuracy = round((classify.accuracy(LinearRegression_classifier, self.testing_set)) * 100,2)


      X_train, y_train = zip(*self.training_set)
      X_train = np.array(pd.DataFrame.from_records(X_train))
      y_train = [(i=='spam')+1 for i in y_train]
      y_train = np.array(y_train)

      X_test, y_test = zip(*self.testing_set)
      X_test = np.array(pd.DataFrame.from_records(X_test))
      y_test = [(i == 'spam') + 1 for i in y_test]
      y_test = np.array(y_test)

      LinearRegression_classifier = LinearRegression()
      LinearRegression_classifier.fit(X_train,y_train)
      R2 = round(LinearRegression_classifier.score(X_test,y_test),2)
      print('Coeffiecient of Determination (R^2):',R2)

      news_ys = []
      spam_ys = []
      for x,y in zip(X_test,y_test):
         y_hat = LinearRegression_classifier.predict(x.reshape(1, -1))[0]
         if y == 1:
            news_ys.append(y_hat)
         else:
            spam_ys.append(y_hat)

      print('')
      print('         News    Not-News')
      print('mean:    ',round(np.mean(news_ys),2),'  ',round(np.mean(spam_ys),2))
      print('median:  ',round(np.median(news_ys),2),'  ', round(np.median(spam_ys),2))
      print('          ',len(news_ys),'    ',len(spam_ys))

      # Plot distribution
      fig = pl.figure(num='Distributions')

      ax1 = fig.add_subplot(121)
      news_ys = sorted(news_ys)
      # bins1 = np.arange(min(news_ys), max(news_ys) + 0.5, 0.5)

      fit_news = stats.norm.pdf(news_ys, np.mean(news_ys), np.std(news_ys))  # this is a fitting indeed
      ax1.plot(news_ys, fit_news)
      ax1.hist(news_ys, color='skyblue', normed=True,lw=1,ec='k'
               # ,bins=bins1
               )  # use this to draw histogram of your data
      ax1.grid(True)
      ax1.set_title('News')

      ax2 = fig.add_subplot(122,sharex=ax1)
      spam_ys = sorted(spam_ys)
      # bins2 = np.arange(min(spam_ys), max(spam_ys) + 0.1, 0.1)

      fit_spam = stats.norm.pdf(spam_ys, np.mean(spam_ys), np.std(spam_ys))  # this is a fitting indeed
      ax2.plot(spam_ys, fit_spam)
      ax2.hist(spam_ys,color='skyblue', normed=True,lw=1,ec='k'
               # ,bins=bins2
               )  # use this to draw histogram of your data
      ax2.set_title('Not-News')
      ax2.grid(True)

      ax1.set_ylim(ax2.get_ylim())
      ax1.set_xlim(ax2.get_xlim())

      # fig.show()  # use may also need add this

      print('------------------------------------------------------------------------')
      print('Naive Bayes:')
      Naivebayes_classifier = NaiveBayesClassifier.train(self.training_set)
      with open(getcwd()+"\\classifiers\\words_as_features\\Naivebayes_classifier.pickle", "wb") as classifier_f:
         pickle.dump(Naivebayes_classifier,classifier_f)
         classifier_f.close()
      Naivebayes_accuracy = round((classify.accuracy(Naivebayes_classifier, self.testing_set)) *100,2)
      self.kappas['kappa_Naivebayes'] = Kappa(Naivebayes_classifier,self.testing_set).kappa_value
      self.accuracies['acc_Naivebayes'] = Naivebayes_accuracy

      if with_print:
         print("Accuracy percent:",Naivebayes_accuracy)
         Naivebayes_classifier.show_most_informative_features(15)

      print('------------------------------------------------------------------------')
      print('Multinomial Naive Bayes:')
      MNB_classifier = SklearnClassifier(MultinomialNB())
      MNB_classifier.train(self.training_set)
      MNB_accuracy = round((classify.accuracy(MNB_classifier, self.testing_set)) * 100,2)
      self.accuracies['acc_MNB'] = MNB_accuracy
      self.kappas['kappa_MNB'] = Kappa(MNB_classifier,self.testing_set).kappa_value

      with open(getcwd()+"\\classifiers\\words_as_features\\MNB_classifier.pickle", "wb") as classifier_f:
         pickle.dump(MNB_classifier,classifier_f)
         classifier_f.close()
      if with_print:
         print("Accuracy percent:",MNB_accuracy)

      print('------------------------------------------------------------------------')
      print('Bernoulli Naive Bayes:')
      BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
      BernoulliNB_classifier.train(self.training_set)
      BernoulliNB_accuracy = round((classify.accuracy(BernoulliNB_classifier, self.testing_set)) * 100,2)
      self.accuracies['acc_BernoulliNB'] = BernoulliNB_accuracy
      self.kappas['kappa_BernoulliNB'] = Kappa(BernoulliNB_classifier,self.testing_set).kappa_value

      with open(getcwd()+"\\classifiers\\words_as_features\\BernoulliNB_classifier.pickle", "wb") as classifier_f:
         pickle.dump(BernoulliNB_classifier,classifier_f)
         classifier_f.close()
      if with_print: print("Accuracy percent:", BernoulliNB_accuracy)


      # Logistic Regression
      print('------------------------------------------------------------------------')
      print('Logistic Regression:')
      LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
      LogisticRegression_classifier.train(self.training_set)
      LogReg_accuracy = round((classify.accuracy(LogisticRegression_classifier, self.testing_set)) * 100,2)
      self.accuracies['acc_LogReg'] = LogReg_accuracy
      self.kappas['kappa_LogReg'] = Kappa(LogisticRegression_classifier,self.testing_set).kappa_value

      with open(getcwd()+"\\classifiers\\words_as_features\\LogisticRegression_classifier.pickle", "wb") as classifier_f:
         pickle.dump(LogisticRegression_classifier,classifier_f)
         classifier_f.close()
      if with_print:
         print("Accuracy percent:",LogReg_accuracy)

      print('------------------------------------------------------------------------')
      print('Stochastic Gradient Descent:')
      SGD_classifier = SklearnClassifier(SGDClassifier())
      SGD_classifier.train(self.training_set)
      SGD_accuracy = round((classify.accuracy(SGD_classifier, self.testing_set)) * 100,2)
      self.accuracies['acc_SGD'] = SGD_accuracy
      self.kappas['kappa_SGD'] = Kappa(SGD_classifier,self.testing_set).kappa_value

      with open(getcwd()+"\\classifiers\\words_as_features\\SGDClassifier_classifier.pickle", "wb") as classifier_f:
         pickle.dump(SGD_classifier,classifier_f)
         classifier_f.close()
      if with_print:
         print("Accuracy percent:",SGD_accuracy)

      print('------------------------------------------------------------------------')
      print('C-Support Vector Machine:')
      SVC_classifier = SklearnClassifier(SVC())
      SVC_classifier.train(self.training_set)
      SVC_accuracy = round((classify.accuracy(SVC_classifier, self.testing_set)) * 100,2)
      self.accuracies['acc_SVC'] = SVC_accuracy
      self.kappas['kappa_SVC'] = Kappa(SVC_classifier,self.testing_set).kappa_value

      with open(getcwd()+"\\classifiers\\words_as_features\\SVC_classifier.pickle", "wb") as classifier_f:
         pickle.dump(SVC_classifier,classifier_f)
         classifier_f.close()
      if with_print:
         print("Accuracy percent:",SVC_accuracy)

      print('------------------------------------------------------------------------')
      print('Linear Support Vector Machine:')
      LinearSVC_classifier = SklearnClassifier(LinearSVC())
      LinearSVC_classifier.train(self.training_set)
      LinearSVC_accuracy = round((classify.accuracy(LinearSVC_classifier, self.testing_set)) * 100,2)
      self.accuracies['acc_LinearSVC'] = LinearSVC_accuracy
      self.kappas['kappa_LinearSVC'] = Kappa(LinearSVC_classifier,self.testing_set).kappa_value

      with open(getcwd()+"\\classifiers\\words_as_features\\LinearSVC_classifier.pickle", "wb") as classifier_f:
         pickle.dump(LinearSVC_classifier,classifier_f)
         classifier_f.close()
      if with_print:
         print("Accuracy percent:",LinearSVC_accuracy)

      # print('------------------------------------------------------------------------')
      # print('Epsilon-Support Vector Machine:')
      # SVR_classifier = SklearnClassifier(SVR())
      # SVR_classifier.train(self.training_set)
      #
      # SVR_accuracy = round((classify.accuracy(SVR_classifier, self.testing_set)) * 100, 2)
      # self.accuracies['SVR'] = SVR_accuracy
      # self.kappas['SVR'] = Kappa(SVR_classifier, self.testing_set).kappa_value
      #
      # with open(getcwd() + "\\classifiers\\words_as_features\\SVR_classifier.pickle", "wb") as classifier_f:
      #    pickle.dump(LinearSVC_classifier, classifier_f)
      #    classifier_f.close()
      # if with_print:
      #    print("Accuracy percent:", SVR_classifier)

      print('------------------------------------------------------------------------')
      print('Multi-layer Perceptron:')
      MLP_Classifier = SklearnClassifier(MLPClassifier(alpha=1))
      MLP_Classifier.train(self.training_set)
      MLP_accuracy = round((classify.accuracy(MLP_Classifier, self.testing_set)) * 100,2)
      self.accuracies['acc_MLP'] = MLP_accuracy
      self.kappas['kappa_MLP'] = Kappa(MLP_Classifier,self.testing_set).kappa_value

      if with_print:
         print("Accuracy percent:",MLP_accuracy)

      print('------------------------------------------------------------------------')
      print('Random Forest:')
      RandomForest_Classifier = SklearnClassifier(RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1))
      RandomForest_Classifier.train(self.training_set)
      RandomForest_accuracy = round((classify.accuracy(RandomForest_Classifier, self.testing_set)) * 100,2)
      self.accuracies['acc_RandomForest'] = RandomForest_accuracy
      self.kappas['kappa_RandomForest'] = Kappa(RandomForest_Classifier,self.testing_set).kappa_value

      if with_print:
         print("Accuracy percent:",RandomForest_accuracy)

      print('------------------------------------------------------------------------')
      print('Adaptive Boosting:')
      AdaBoost_Classifier = SklearnClassifier(AdaBoostClassifier())
      AdaBoost_Classifier.train(np.array(self.training_set))
      AdaBoost_accuracy = round((classify.accuracy(AdaBoost_Classifier, self.testing_set)) * 100,2)
      self.accuracies['acc_AdaBoost'] = AdaBoost_accuracy
      self.kappas['kappa_AdaBoost'] = Kappa(AdaBoost_Classifier,self.testing_set).kappa_value

      if with_print:
         print("Accuracy percent:",AdaBoost_accuracy)

      print('------------------------------------------------------------------------')
      print('Voted Classifier:')
      voted_classifier = VoteClassifier(Naivebayes_classifier,
                                        # SVR_classifier,
                                        MLP_Classifier,
                                        RandomForest_Classifier,
                                        # QDA_Classifier,
                                        AdaBoost_Classifier,
                                        LinearSVC_classifier,
                                        SGD_classifier,
                                        MNB_classifier,
                                        BernoulliNB_classifier,
                                        LogisticRegression_classifier)

      Voted_accuracy = round((classify.accuracy(voted_classifier, self.testing_set)) * 100,2)
      self.accuracies['acc_Voted'] = Voted_accuracy
      self.kappas['kappa_voted'] = Kappa(voted_classifier,self.testing_set).kappa_value
      if with_print: print("Accuracy percent:", Voted_accuracy)
      print('------------------------------------------------------------------------')


      # Saving results to file

      results = {**self.accuracies,**self.kappas}
      results['timestamp'] = time.strftime('%Y%m%d_%H:%M:%S',time.gmtime(time.time()))
      results['R2'] = R2

      df = pd.DataFrame().from_csv(getcwd() + "\\classifiers\\words_as_features\\confs.csv",sep=";")
      df = df.append(pd.Series(results),ignore_index=True)
      df.to_csv(getcwd() + "\\classifiers\\words_as_features\\confs.csv",sep=";")
      # print(df)


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

