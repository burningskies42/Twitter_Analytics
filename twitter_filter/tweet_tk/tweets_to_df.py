from pandas import DataFrame,Series,concat,set_option,read_pickle
from json import loads,load

set_option('display.float_format', lambda x: '%.3f' % x)

# converts from json to tweet DB
# flattens out all user columns
def tweet_json_to_df(fileName,flatten_user=True):
    # try:
    with open(fileName, 'r') as file:
        lines = [line.rstrip('\n') for line in file]

    # remove empty lines
    lst = []
    for i in range(len(lines)):
        if i % 2 == 0:
            line = lines[i]

            json_line = loads(line)

            lst.append(json_line)

    df = DataFrame(lst)

    if flatten_user:
        users_df = df['user'].apply(Series)
        users_df.columns = 'user_' + users_df.columns
        df = concat([df.drop(['user'], axis=1), users_df], axis=1)

    df.set_index('id', inplace=True)
    return df
    # except Exception as e:
    #     print('unsuccessful convertsion:',e)

# converts from pickle to tweet DB
# also flattens out all user columns
def tweet_pickle_to_df(pth,flatten_user=True):
    df = read_pickle(pth)
    if flatten_user:
        users_df = df['user'].apply(Series)
        users_df.columns = 'user_' + users_df.columns
        df = concat([df.drop(['user'], axis=1), users_df ], axis=1)
    DataFrame.set_index(df,'id',inplace=True)
    return df
