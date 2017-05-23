from labeled_featureset_builder import open_and_join
from sklearn import preprocessing,model_selection
from sklearn import svm
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from labeled_featureset_builder import fetch_tweets_by_ids,fetch_tweet
from feature_tk.features import tweets_to_featureset,single_tweet_features
import statsmodels.api as sm

# open_and_join('labels/Amazon_labeled_tweets.csv',True)

df = pd.DataFrame.from_csv('labeled_featureset.csv',sep=';')
df = df[df['label']!=3]
df.dropna(axis=0,how='any',inplace=True)
print(len(df))

#
# df = df.drop(['has_pronoun','count_upper','has_hashtag','friends_cnt'],axis=1)
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
clf = LinearRegression(n_jobs=-1)
# clf = svm.SVC()
clf.fit(X_train, y_train)
confidence = clf.score(X_test, y_test)
print('confidence:',confidence)

tester = fetch_tweets_by_ids([866661103203360768,866668601419456512,866668290290184192,866630898715947009])
tester = tweets_to_featureset(tester)

test = np.array(tester)

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()

print(est2.summary())

# print(clf.predict(test))
