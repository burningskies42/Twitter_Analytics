from labeled_featureset_builder import open_and_join
import pandas as pd
import numpy as np

from sklearn import preprocessing, model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVR
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LinearRegression,LogisticRegression

df = open_and_join('C:/Users/Leon/Documents/Masterarbeit/Python/twitter_filter/labels/amazon_labeled_tweets.csv',with_sentiment=False)

all_words = {}

for tw in df['words']:
   for word in tw:
      if word in all_words.keys():
         all_words[word] +=1
      else:
         all_words[word] = 1


# for word in sorted(all_words, key=all_words.get,reverse=True):
#    print(word, all_words[word])

def find_features(document, word_features):
   words = set(document)
   features = {}
   for w in word_features:
      features[w] = (w in words)

   return features


feature_words = sorted(all_words, key=all_words.get,reverse=True)[:3000]

WAF_df = pd.DataFrame(index=df.index,columns=feature_words)
# WAF_d

for i,row in df.iterrows():
   # print(row['words'])
   # print(feature_words)
   features_set = find_features(document=row['words'],word_features=feature_words)
   features_set = pd.Series(features_set)

   WAF_df.loc[i] = features_set

df = pd.concat([df,WAF_df],axis=1)
df = df.drop(['words','words_no_url'],axis=1)

X = np.array(df.drop(['label'], 1))
# X_scaled = preprocessing.scale(X)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1)

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
   MLPClassifier(alpha=1,solver='sgd',activation='tanh'),
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

for name, clf in zip(names, classifiers):
   clf.fit(X_train, y_train)
   print(name + ' confidence:', clf.score(X_test, y_test))

   kappa_matrix = {}
   kappa_matrix['news'] = {}
   kappa_matrix['news']['true'] = 0
   kappa_matrix['news']['false'] = 0

   kappa_matrix['spam'] = {}
   kappa_matrix['spam']['true'] = 0
   kappa_matrix['spam']['false'] = 0

   for X,y in zip(X_test,y_test):
      alg_class = clf.predict(X.reshape(1, -1))

      # Actully news
      if y == 1:
         # algo classified correctly (a)
         if alg_class <= 1.5 :
            kappa_matrix['news']['true'] += 1

         # algo misclassified 'news' as 'spam' (b)
         else:
            kappa_matrix['spam']['false'] += 1

      # true classification is 'spam'
      elif y == 2:
         # # algo classified correctly (d)
         if alg_class > 1.5:
            kappa_matrix['spam']['true'] += 1

         # algo misclassified 'spam' as 'news' (c)
         else:
            kappa_matrix['news']['false'] += 1

   a = kappa_matrix['news']['true']
   b = kappa_matrix['spam']['false']
   c = kappa_matrix['news']['false']
   d = kappa_matrix['spam']['true']
   abcd = a + b + c + d
   p_zero = (a + d) / abcd
   p_news = ((a + b) / abcd) * ((a + c) / abcd)
   p_spam = ((c + d) / abcd) * ((b + d) / abcd)
   p_e = p_news + p_spam
   # print(p_news,p_spam)
   kappa_value = (p_zero - p_e) / (1 - p_e)
   print("Cohen's Kappa:", kappa_value)
   # print(a,b,c,d)
   print('========================================')

