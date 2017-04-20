import pandas as pd
import quandl
from statistics import mean
import numpy as np
from sklearn import svm, preprocessing, cross_validation

# style.use('ggplot')
# ax1 = plt.gca()
# ax2 = ax1.twinx()

api_key = open('quandl_key.txt','r').read()

## Get functions::

def moving_average(values):
    return mean(values)

def morgage_30y():
    df = quandl.get('FMAC/MORTG',trim_start = '1975-01-01',authtoken = api_key)
    df['Value'] = (df['Value'] - df['Value'][0]) / df['Value'] * 100.0
    df = df.resample('D').mean()
    df = df.resample('M').mean()
    df.columns = ['M30']
    return df

def HPI_Benchmark():
    df = quandl.get('FMAC/HPI_USA', authtoken = api_key)
    df['Value'] = (df['Value'] - df['Value'][0]) / df['Value'][0] * 100.0
    df.columns.values[0] = 'US_HPI'
    return df

def sp500_data():
    df = quandl.get("YAHOO/INDEX_GSPC", trim_start="1975-01-01", authtoken=api_key)
    df["Adjusted Close"] = (df["Adjusted Close"]-df["Adjusted Close"][0]) / df["Adjusted Close"][0] * 100.0
    df=df.resample('M').mean()
    df.rename(columns={'Adjusted Close':'sp500'}, inplace=True)
    df = df['sp500']
    return df

def gdp_data():
    df = quandl.get("BCB/4385", trim_start="1975-01-01", authtoken=api_key)
    df["Value"] = (df["Value"]-df["Value"][0]) / df["Value"][0] * 100.0
    df=df.resample('M').mean()
    df.rename(columns={'Value':'GDP'}, inplace=True)
    df = df['GDP']
    return df

def us_unemployment():
    df = quandl.get("ECPI/JOB_G", trim_start="1975-01-01", authtoken=api_key)
    df["Unemployment Rate"] = (df["Unemployment Rate"]-df["Unemployment Rate"][0]) / df["Unemployment Rate"][0] * 100.0
    df=df.resample('1D').mean()
    df=df.resample('M').mean()
    return df

def create_labels(cur_hpi, fut_hpi):
    if fut_hpi > cur_hpi:
        return 1
    else:
        return 0

# # compiling and pickling data::
# sp500 = sp500_data()
# us_gdp = gdp_data()
# us_unemployment = us_unemployment()
#
# hpi_data = pd.read_pickle('all_states.pickle')
# m30 = morgage_30y()
# benchmark = HPI_Benchmark()
# HPI = hpi_data.join([benchmark, m30, us_unemployment, us_gdp, sp500])
#
# HPI.dropna(inplace = True)
# HPI.to_pickle('hpi_usw.pickle')

housing_data = pd.read_pickle('hpi_usw.pickle')
housing_data = housing_data.pct_change()

housing_data.replace([np.inf, -np.inf], np.nan,inplace = True)
housing_data.dropna(inplace = True)

housing_data['US_HPI_future']  = housing_data['US_HPI'].shift(-1)
housing_data.dropna(inplace = True)

housing_data['label'] = list(map(create_labels,housing_data['US_HPI'],housing_data['US_HPI_future']))

X = np.array(housing_data.drop(['label','US_HPI_future'],1))
X = preprocessing.scale(X)

Y = np.array(housing_data['label'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size=0.2)

clf = svm.SVC(kernel='linear')
clf.fit(X_train,Y_train)
print(clf.score(X_test,Y_test))