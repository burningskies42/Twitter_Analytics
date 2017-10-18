import pandas as pd
from os import getcwd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection
import numpy as np
from statsmodels.formula.api import ols,logit


pth = getcwd()+'/words_as_features/weighted_confs.csv'

df = pd.DataFrame.from_csv(pth,sep=';')
df.set_index('time_stamp',inplace=True)


# lst= list(set(df['Name']))
# cls_dct = {}
# for c in lst:
#    cls_dct[c] = lst.index(c)
#
# for key,val in cls_dct.items():
#    df['Name'].replace(to_replace=key,value=val,inplace=True)


# X = np.array(df.drop(['News_F1'], 1))
# X = preprocessing.scale(X)
# y = np.array(df['News_F1'])
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

dummies = pd.get_dummies(df['Name'])
dummies = pd.concat([df,dummies],axis=1)
dummies.drop(['Name'],axis=1)
label = 'News_F1'
vars = dummies.columns.values

large_formula = """News_F1 ~ 
LogisticRegression +
NaiveBayes +
BernoulliNB +
MultinomialNB +
LinearSVC +        
SGD +
MLP +
RandomForest +
AdaBoost +
Vote"""

# Dummies for clf
# model = ols(large_formula, dummies).fit()
print(df.describe())
model = ols('News_F1 ~ C(Name) ', dummies).fit()
print(model.summary())


'''
=== SVM KERNELS ====== SVM KERNELS ====== SVM KERNELS ====== SVM KERNELS ====== SVM KERNELS ====== SVM KERNELS ====== SVM KERNELS ====== SVM KERNELS ===
'''

pth = getcwd()+'/words_as_features/kernels_weighted_confs.csv'

df = pd.DataFrame.from_csv(pth,sep=';')
df.set_index('time_stamp',inplace=True)

df = df[['Kernel','Train_News','Train_Spam','gamma','C','Accuracy','Kappa','News_F1']]
df=df.rename(columns = {'C':'cost'})

# dummies = pd.get_dummies(df['Kernel'])
# df = pd.concat([df,dummies],axis=1)

df = df[df['gamma'] != 'auto']
df['gamma'] = df['gamma'].astype('float64')


model = ols('News_F1 ~ C(Kernel) + np.log(gamma) + np.log(cost)', df).fit()
# print(dummies.describe())
print(model.summary())