from os import listdir
from easygui import diropenbox
import tweet_tk.tweets_to_df as tweets_to_df
from pandas import DataFrame
from time import strftime
from pickle import dump

# open path containing all jsion files
path = diropenbox()
datasets = [f for f in listdir(path) if f.split('.')[1] == 'json']
agg_dataset = DataFrame()

# load all jsons into agg_dataset
for f in datasets:
    curr_path = path+'\\'+f
    print(curr_path)
    df = tweets_to_df.tweet_json_to_df(curr_path)
    print('loaded',f,'size:',len(df))
    agg_dataset = agg_dataset.append(df)

print('\ndone aggregating. total size:',len(agg_dataset))

# drop aggregated dataset into pickle
agg_file_name = path+'/agg_' + strftime("%Y_%m_%d_%H_%M_%S") + '.pickle'

save_file = open(agg_file_name,'wb')
dump(agg_dataset,save_file)
save_file.close()
# agg_dataset




