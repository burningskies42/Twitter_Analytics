import pandas as pd
import numpy as np
import json
pd.set_option('display.float_format', lambda x: '%.3f' % x)


def tweet_json_to_df(fileName):
    data = []
    cnt = 0
    try:
        with open(fileName) as data_file:
            for line in data_file:
                cnt += 1
                line = json.loads(line)
                # print(cnt,line)
                data.append(line)

        df = pd.DataFrame(data)

        # df = pd.concat([df.drop(['user'], axis=1), df['user'].apply(pd.Series)], axis=1)

        # df.set_index('id',inplace=True)
        return df
    except Exception as e:
        print(e)

def build_user_df(df):
    users = pd.DataFrame()
    for i in df['user']:
        if users.empty:
            users = pd.DataFrame([i])
        else:
            users = users.append([i])

    users.drop_duplicates(inplace=True)
    users.set_index('id', inplace=True)
    return users

# converts from pickle to tweet DB
# also flattens out all user columns
def tweet_pickle_to_df(pth):
    df = pd.read_pickle(pth)
    users_df = df['user'].apply(pd.Series)
    users_df.columns = 'user_' + users_df.columns
    df = pd.concat([df.drop(['user'], axis=1), users_df ], axis=1)
    pd.DataFrame.reset_index(df,'id_str',inplace=True)
    return df


# df = tweet_json_to_df('amazon_db.json')





