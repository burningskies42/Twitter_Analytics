from labeled_featureset_builder import open_and_join
from sklearn import preprocessing,model_selection
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from labeled_featureset_builder import fetch_tweets_by_ids,fetch_tweet
from feature_tk.features import tweets_to_featureset,single_tweet_features
import statsmodels.api as sm
import pickle
import easygui

def build_and_classify():
   # Build the feature_set - Only necessary once
   f_path = easygui.fileopenbox()  #   'labels/Amazon_labeled_tweets.csv'
   open_and_join(f_path,True,with_sentiment = False)
   #
   df = pd.DataFrame.from_csv('labeled_featureset.csv',sep=';')
   df = df[df['label']!=3]
   df.dropna(axis=0,how='any',inplace=True)
   print(len(df))

   #
   # df = df.drop(['has_pronoun','count_upper','has_hashtag','friends_cnt'],axis=1)
   X = np.array(df.drop(['label'], 1))
   y = np.array(df['label'])

   # Not needed for RF
   # PP_X = preprocessing.scale(X)

   PP_X = X

   X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

   # Preprocessed
   pp_X_train, pp_X_test, y_train, y_test = model_selection.train_test_split(PP_X, y, test_size=0.2)

   lin_clf = LinearRegression(n_jobs=-1)
   svm_clf = svm.SVC()
   RF_clf = RandomForestClassifier(n_estimators=50)

   lin_clf.fit(pp_X_train, y_train)
   svm_clf.fit(pp_X_train, y_train)
   RF_clf.fit(X_train, y_train)

   lin_confidence = lin_clf.score(pp_X_test, y_test)
   svm_confidence = svm_clf.score(pp_X_test, y_test)
   RF_confidence = RF_clf.score(X_test, y_test)
   print('\n--------------------------------------------\n'+
      'Linear Regression confidence:',lin_confidence,'\n'+
      'SVM confidence:', svm_confidence,'\n'+
      'Random Forest confidence:', RF_confidence,'\n'+
      '--------------------------------------------\n')


   # save the classifiers
   with open('lin_clf.pkl', 'wb') as fid:
      pickle.dump(lin_clf, fid)
      fid.close()

   with open('svm_clf.pkl', 'wb') as fid:
      pickle.dump(svm_clf, fid)
      fid.close()

   with open('RF_clf.pkl', 'wb') as fid:
      pickle.dump(RF_clf, fid)
      fid.close()

# Uncomment when training again, otherwise use existing classifier
build_and_classify()

# load it again
with open('lin_clf.pkl', 'rb') as fid:
   lin_clf = pickle.load(fid)
   fid.close()

with open('svm_clf.pkl', 'rb') as fid:
   svm_clf = pickle.load(fid)
   fid.close()

with open('RF_clf.pkl', 'rb') as fid:
   RF_clf = pickle.load(fid)
   fid.close()

tester = fetch_tweets_by_ids([866661103203360768,866668601419456512,866668290290184192,866630898715947009])
tester = tweets_to_featureset(tester,with_sentiment=False)

test = np.array(tester)

# # Regression print-out:
# X2 = sm.add_constant(X)
# est = sm.OLS(y, X2)
# est2 = est.fit()
#
# print(est2.summary())

print('Linear:',lin_clf.predict(test))
print('SVM:',svm_clf.predict(test))
print('Random forest',RF_clf.predict(test))
