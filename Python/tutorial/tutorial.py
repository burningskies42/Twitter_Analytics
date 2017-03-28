import pandas as pd
import quandl
import pickle
import tensorflow
import matplotlib.pyplot as plt
from matplotlib import style
from pip._vendor import colorama

style.use('fivethirtyeight')

def quandl_key():
    api_key=open('quandl_key.txt','r').read()
    return api_key

api_key = quandl_key()

def states_list():
    all_states = pd.read_html('https://simple.wikipedia.org/wiki/List_of_U.S._states')
    return all_states[0][0][1:]

def grab_init_state_list():
    main_df = pd.DataFrame()
    s_list = states_list()

    for abbv in s_list:
        query = 'FMAC/HPI_' + str(abbv)
        df = quandl.get(query,authtoken = api_key)
        df.columns = [str(abbv)]

        #Percent change:
        # df = df.pct_change()
        df[abbv] = (df[abbv] - df[abbv][0]) / df[abbv][0] * 100.0


        # print(df.head())
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

    pickle_out = open('all_states.pickle','wb')
    pickle.dump(main_df,pickle_out)
    pickle_out.close()

# grab_init_state_list()

def HPI_Benchmanrk():
    df = quandl.get('FMAC/HPI_USA', authtoken=api_key)
    df.columns = ['United States']
    return df

hpi_data = pd.read_pickle('all_states.pickle')
hpi_avg = HPI_Benchmanrk()

# fig = plt.figure()
# ax1 = plt.subplot2grid((1,1),(0,0))
#
# hpi_data.plot(ax = ax1)
# hpi_avg.plot(ax = ax1, color = 'k', linewidth = 10)
#
# plt.legend().remove()
# print(hpi_avg)
# plt.show()

hpi_corr = hpi_data.corr()
hpi_corr.to_csv('hpi_corr.csv')
# print(hpi_corr)