import pandas as pd
import quandl
import pickle

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
        # print(df.head())
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df)

    pickle_out = open('all_states.pickle','wb')
    pickle.dump(main_df,pickle_out)
    pickle_out.close()

# grab_init_state_list()
pickle_in = open('all_states.pickle','rb')
HPI_DATA = pickle.load(pickle_in)
print('wrote to HPI_DATA')

HPI_DATA.to_pickle('pd_pickle.pickle')
hpi_data2 = pd.read_pickle('pd_pickle.pickle')
print(hpi_data2.head())