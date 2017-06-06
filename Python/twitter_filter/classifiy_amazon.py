from labeled_featureset_builder import open_and_join
from sklearn import preprocessing, model_selection
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import neighbors
from sklearn.cluster import KMeans, MeanShift
import os

from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import export_graphviz
import pandas as pd
import numpy as np
from labeled_featureset_builder import fetch_tweets_by_ids, fetch_tweet
from feature_tk.features import tweets_to_featureset, single_tweet_features
import pickle
import pydot
import easygui


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVR
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
   LinearRegression(n_jobs=-1),
   LogisticRegression(n_jobs=-1),
   KNeighborsClassifier(3),
   SVC(gamma=2, C=1),
   # SVC(kernel="linear", C=0.025),
   LinearSVR(C=0.025),
   # GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
   DecisionTreeClassifier(max_depth=5),
   RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
   MLPClassifier(alpha=1),
   AdaBoostClassifier(),
   GaussianNB(),
   QuadraticDiscriminantAnalysis()]

names = ["Linear Regression",
         "Logistic Regression",
         "Nearest Neighbors",
         "RBF SVM",
         "Linear SVM",
         # "Gaussian Process",
         "Decision Tree",
         "Random Forest",
         "Neural Net",
         "AdaBoost",
         "Naive Bayes",
         "QDA"]


def build_and_classify():
   # # Build the feature_set - Only necessary once
   f_path = easygui.fileopenbox()  #   'labels/Amazon_labeled_tweets.csv'
   df = open_and_join(f_path,True,with_sentiment = False,with_timing=True)
   # df = pd.DataFrame.from_csv('labeled_featureset.csv', sep=';')
   df = df[df['label'] != 3]
   df.dropna(axis=0, how='any', inplace=True)
   print('Sample size:', len(df))

   #
   # df = df.drop(['has_pronoun','count_upper','has_hashtag','friends_cnt'],axis=1)
   X = np.array(df.drop(['words','words_no_url','label'], 1))
   y = np.array(df['label'])

   # Not needed for RF
   # PP_X = preprocessing.scale(X)

   PP_X = X

   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.9)

   # # Manual division
   # X_train = X[:-2000]
   # X_test = X[-2000:]
   # y_train = y[:-2000]
   # y_test = y[-2000:]

   # Preprocessed
   # pp_X_train, pp_X_test, y_train, y_test = model_selection.train_test_split(PP_X, y, test_size=0.2)

   tester = fetch_tweets_by_ids([866661103203360768, 866668601419456512, 866668290290184192, 866630898715947009])
   tester = tweets_to_featureset(tester, with_sentiment=False)
   tester.drop(['words', 'words_no_url'], axis=1, inplace=True)

   test = np.array(tester)

   for name, clf in zip(names, classifiers):
      clf.fit(X_train, y_train)
      print(name+' confidence:', clf.score(X_test, y_test))

      f_name = os.getcwd()+'\\classifiers\\' + name.replace(' ','_') + '_clf.pkl'

      if not os.path.exists(f_name):
         open(f_name, 'a').close()
         print('created',f_name)

      with open( f_name, 'wb') as fid:
         pickle.dump(clf, fid)
         fid.close()

   for name, clf in zip(names, classifiers):
      print(name,clf.predict(test))

   print('------------------------------------')

   # for filename in os.listdir('classifiers'):
   #    with open(filename, 'rb') as fid:
   #       f_name = filename.split('_clf')[0]
   #       cls = pickle.loads(fid)
   #
   #       cls_dict.append((f_name,cls))
   #       print('loaded',filename)
   #       fid.close()

   # print('------------------------------------')

   # import statsmodels.api as sm
   #
   # # Note the swap of X and y
   #
   # # Regression print-out:
   # X2 = sm.add_constant(X)
   # est = sm.OLS(y, X2)
   # est2 = est.fit()

   # print(est2.summary())


   # Print out all trees in forest to jpg
   i = 0

   RF_clf_f = open('classifiers\\Random_Forest_clf.pkl','rb')
   RF_clf = pickle.load(RF_clf_f)
   RF_clf_f.close()

   for tree in RF_clf:
      i += 1
      fn = 'trees\\dtree' + str(i) + '.dot'
      dotfile = open(fn, 'w')

      export_graphviz(tree, feature_names=df.columns.values, filled=True, rounded=True, out_file=dotfile)
      dotfile.close()

      (graph,) = pydot.graph_from_dot_file(fn)
      print('exported', fn)
      graph.write_png('trees\\dtree' + str(i) + '.png')


# Uncomment when training again, otherwise use existing classifier
build_and_classify()

