import pandas as pd
import numpy
import emoji

import json
from pprint import pprint

def tweetJson2dataFrame(fileName):
    data = []

    with open(fileName) as data_file:
        for line in data_file:
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    df.set_index('id',inplace=True)
    return df


df = tweetJson2dataFrame('amazon_db.json')
# df.to_csv('amazon.csv',sep=';')


users = pd.DataFrame()
# print(type(users))

for entry in df['user']:
    # if users.empty:
    #     df.columns = entry.keys()

    user = list(entry.values())
    users = users.append(user)

# print(users)

