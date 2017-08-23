import pandas as pd
from os import getcwd
import pickle
import random

import optunity
import optunity.metrics
import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

global_score = None

def find_features(document, word_features):
   words = set(document)
   features = {}
   for w in word_features:
      features[w] = (w in words)

   return features

with open(getcwd() + "\\classifiers\\words_as_features\\Documents.pickle", "rb") as fid:
   documents = pickle.load(fid)
   fid.close()

'''
1 - News
2 - Not News
'''
all_words = {}

documents = [(words, 1) if (label == 'news') else (words, 0) for (words, label) in documents]
for tweet in documents:
   for word in tweet[0]:
      if word.lower() in all_words.keys():
         all_words[word.lower()] += 1
      else:
         all_words[word.lower()] = 1

# Get the 5000 most popular words
num_features = 3000
word_features = sorted(all_words.items(), key=lambda x: x[1], reverse=True)[10:(num_features + 10)]
word_features = [w[0] for w in word_features]

random.shuffle(documents)
featuresets = [(find_features(rev, word_features), category) for (rev, category) in documents]


def train_svm(data, labels, kernel, C, gamma, degree, coef0):
   """A generic SVM training function, with arguments based on the chosen kernel."""
   if kernel == 'linear':
      model = SVC(kernel=kernel, C=C)
   elif kernel == 'poly':
      model = SVC(kernel=kernel, C=C, degree=degree, coef0=coef0)
   elif kernel == 'rbf':
      model = SVC(kernel=kernel, C=C, gamma=gamma)
   else:
      raise ValueError("Unknown kernel function: %s" % kernel)
   model.fit(data, labels)
   return model

search = {'algorithm': {'k-nn': {'n_neighbors': [1, 5]},
                        'SVM': {'kernel': {'linear': {'C': [0, 2]},
                                           'rbf': {'gamma': [0, 1], 'C': [0, 10]},
                                           'poly': {'degree': [2, 5], 'C': [0, 50], 'coef0': [0, 1]}
                                           }
                                }
         }
   }




data = [list(i[0].values()) for i in featuresets]
new_data = []

for set in data:
   set = [1 if x else 0 for x in set]
   new_data.append(set)

data = np.array(new_data)

labels = np.array([True if y == 1 else False for y in [i[1] for i in featuresets]])


@optunity.cross_validated(x=data, y=labels, num_folds=5)
def performance(x_train, y_train, x_test, y_test,
                algorithm, n_neighbors=None, n_estimators=None, max_features=None,
                kernel=None, C=None, gamma=None, degree=None, coef0=None):

    # fit the model
    if algorithm == 'k-nn':
        model = KNeighborsClassifier(n_neighbors=int(n_neighbors))
        model.fit(x_train, y_train)
    elif algorithm == 'SVM':
        model = train_svm(x_train, y_train, kernel, C, gamma, degree, coef0)
    elif algorithm == 'naive-bayes':
        model = GaussianNB()
        model.fit(x_train, y_train)
    elif algorithm == 'random-forest':
        model = RandomForestClassifier(n_estimators=int(n_estimators),
                                       max_features=int(max_features))
        model.fit(x_train, y_train)
    else:
        raise ValueError('Unknown algorithm: %s' % algorithm)

    print(algorithm,'========')
    print('kernel:',kernel)
    print('gamma:',gamma)
    print('C:',C)
    print('coef0:', coef0)
    print('degree:', degree)
    print('score:',model.score(X=x_test, y=y_test),'\n')
    global_score = model.score(X=x_test, y=y_test)

    # predict the test set
    if algorithm == 'SVM':
        predictions = model.decision_function(x_test)
    else:
        predictions = model.predict_proba(x_test)[:, 1]

    return optunity.metrics.roc_auc(y_test, predictions, positive=True)

'''

optimal_configuration, info, _ = optunity.maximize_structured(performance,
                                                              search_space=search,
                                                              num_evals=10)

print('========')
print()
print(optimal_configuration)
print(info.optimum)
print('========')

solution = dict([(k, v) for k, v in optimal_configuration.items() if v is not None])
print('Solution\n========')
print("\n".join(map(lambda x: "%s \t %s" % (x[0], str(x[1])), solution.items())))
'''

df = pd.DataFrame(['coef','C','degree','score'])

for coef in range(0, 1):
   for d in range(2,5):
      for c in range(1, 50):
         per = performance(algorithm='SVM', kernel='poly',C=c,degree=d,coef0=coef)
         df = df.append(pd.Series({'coef':coef,
                                   'C':c,
                                   'degree':d,
                                   'score':global_score
                                   }),ignore_index=True)


df.to_csv('poly_kernel_analysis',sep=';')