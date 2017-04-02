import quandl
import pandas as pd
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

def quandl_key():
    api_key = pd.read_pickle('quandl_key.pickle')
    return api_key

def getTable(forecast_col):
    df = quandl.get('WIKI/GOOGL',authtoken = quandl_key())
    df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
    df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100
    df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100
    df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
    df.fillna(-99999, inplace=True)

    forecast_out = int(math.ceil(0.01 * len(df)))
    df['label'] = df[forecast_col].shift(-forecast_out)
    df.dropna(inplace=True)
    return df

df = getTable(forecast_col = 'Adj. Close')

X = np.array(df.drop(['label'],1))
print(X)



