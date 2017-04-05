import quandl, math
import pandas as pd,numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression

def quandl_key():
    api_key = pd.read_pickle('quandl_key.pickle')
    return api_key


forecast_col = 'Adj. Close'
df = quandl.get('WIKI/GOOGL',authtoken = quandl_key())
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)


X = np.array(df.drop(['label'],1))
y = np.array(df['label'])

X = preprocessing.scale(X)

y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

clf = LinearRegression(n_jobs=-1)
# clf = svm.SVR(kernel='poly')
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)