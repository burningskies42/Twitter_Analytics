import pandas as pd
from sklearn.linear_model import LinearRegression
# from linearmodels.iv import IV2SLS
import numpy as np
import math

df = pd.DataFrame.from_csv('C:/Twitter_Analytics/Python/twitter_filter/classifiers/words_as_features/complete/all_weighted_confs.csv',sep=';')
df.set_index('time_stamp',inplace=True)


df['Total_size'] = df['Train_News'] + df['Train_Spam'] + df['Test_News'] + df['Test_Spam']
# df = df[['rauc','feature_cnt','Total_size']]
# df['feature_cnt'] = [math.log(n) for n in df['feature_cnt']]
# df['feature_cnt2'] = [n^2 for n in df['feature_cnt']]

# print(df.columns.values)

# clf = LinearRegression().fit(X=np.array(df[['Total_size','feature_cnt']]),y=np.array(df['rauc']))

clf = LinearRegression().fit(X=np.array(df[['Total_size']]),y=np.array(df['duration']))

iv = df[['Total_size']]* clf.coef_[0]
print(iv)

# print(clf.coef_)
