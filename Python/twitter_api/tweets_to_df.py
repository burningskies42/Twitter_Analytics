import pandas as pd
import json
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# converts from json to tweet DB
# flattens out all user columns
def tweet_json_to_df(fileName,flatten_user=True):
    try:
        with open(fileName, 'r') as file:
            lines = [line.rstrip('\n') for line in file]

        # remove empty lines
        lst = []
        for i in range(len(lines)):
            if i % 2 == 0:
                lst.append(json.loads(lines[i]))

        df = pd.DataFrame(lst)

        if flatten_user:
            users_df = df['user'].apply(pd.Series)
            users_df.columns = 'user_' + users_df.columns
            df = pd.concat([df.drop(['user'], axis=1), users_df], axis=1)

        df.set_index('id', inplace=True)
        return df
    except Exception as e:
        print('unsuccessful convertsion:',e)

# converts from pickle to tweet DB
# also flattens out all user columns
def tweet_pickle_to_df(pth,flatten_user=True):
    df = pd.read_pickle(pth)
    if flatten_user:
        users_df = df['user'].apply(pd.Series)
        users_df.columns = 'user_' + users_df.columns
        df = pd.concat([df.drop(['user'], axis=1), users_df ], axis=1)
    pd.DataFrame.set_index(df,'id',inplace=True)
    return df
