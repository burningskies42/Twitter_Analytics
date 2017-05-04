# Tweeter toolkit contains all auxilary functions
from tweet_tk import *

# Clear console and print greeting
system('CLS')
print('''
                ***********************************************************************
                *                                                                     *
                *              Feature Set Builder for Twitter Datasets               *
                *                              Version 0.1                            *
                *             Author: Leon Edelmann        Copyright 2017 (c)         *
                *                                                                     *
                ***********************************************************************
''')

print('loaded toolkit packages ...')

# open target tweets dataset
default_path = getcwd()+'\captured_tweets\*.JSON'
dataset_path = easygui.fileopenbox(default=default_path,filetypes=[["*.pickle", "Binary files"]])
if dataset_path == None:
    quit()

# Extract dataset name from path
file_name = dataset_path.split('\\')[len(dataset_path.split('\\'))-1].split('_dataset.json')[0]
# print(file_name)

def msg_feature_df(df):
    df_msg = pd.DataFrame()
    df_msg['id'] = df.index
    df_msg.set_index('id', inplace=True)

    start = time()
    df_msg['words'] = df['text'].apply(lambda x : word_tokenize(x))
    dur = time() - start
    print('tokenize words:',dur)

    start = time()
    df_msg['words_no_url'] = df['text'].apply(lambda x : clear_urls(x))
    dur = time() - start
    print('clear urls:', dur)

    start = time()
    df_msg['duplicate'] = df_msg['words_no_url'].duplicated(keep=False)
    dur = time() - start
    print('duplicates:', dur)

    start = time()
    bot_suspects = set(df['user_id'][df_msg['duplicate']==True])
    add_suspects(bot_suspects)
    dur = time() - start
    print('add_suspects:', dur)


    # start = time()
    # df_msg['similiarity'] = similar(df_msg['words_no_url'])
    # print(similar(df_msg['words_no_url']))
    # dur = time() - start
    # print('similiarity:', dur)

    start = time()
    stop_words = set(stopwords.words('english'))
    df_msg['words'] = df_msg['words'].apply(lambda x : [w for w in x if w.lower() not in stop_words])
    dur = time() - start
    print('filter out stop words:',dur)

    start = time()
    df_msg['len_characters'] = df['text'].apply(lambda x : len(x))
    dur = time() - start
    print('len_characters:', dur)

    start = time()
    df_msg['num_words'] = df_msg['words'].apply(lambda x : len(x))
    dur = time() - start
    print('num_words:', dur)

    start = time()
    df_msg['has_question_mark'] = df['text'].apply(lambda x : x.find('?') != -1)
    dur = time() - start
    print('has_question_mark:', dur)

    start = time()
    df_msg['has_exclamation_mark'] = df['text'].apply(lambda x : x.find('!') != -1)
    dur = time() - start
    print('has_exclamation_mark:', dur)

    start = time()
    df_msg['has_multi_quest_exclam'] = df['text'].apply(lambda x : (x.count('?') > 1 or x.count('!') > 1))
    dur = time() - start
    print('has_multi_quest_exclam:', dur)

    start = time()
    df_msg['emotji_sent_score'] = df['text'].apply(lambda x : emoticons_score(x))
    dur = time() - start
    print('emotji_sent_score:', dur)

    start = time()
    df_msg['has_pronoun'] = df['text'].apply(lambda x : has_pronoun(x))
    dur = time() - start
    print('has_pronoun:', dur)

    start = time()
    df_msg['count_upper'] = df['text'].apply(lambda x : count_upper(x))
    dur = time() - start
    print('count_upper:', dur)

    start = time()
    df_msg['has_hashtag'] = df['text'].apply(lambda x: x.find('#') != -1)
    dur = time() - start
    print('has_hashtag:', dur)

    start = time()
    df_msg['urls'] = df['entities'].apply(lambda x: get_urls(x))
    dur = time() - start
    print('urls:', dur)

    start = time()
    sentm = df['text'].apply(lambda x: sentiment(x))
    sentm = sentm.apply(pd.Series)
    sentm.columns = ['class', 'conf']
    df_msg['senitment'] = sentm['class']
    df_msg['senitment_conf'] = sentm['conf']
    dur = time() - start
    print('senitment:', dur)

    return df_msg

def usr_feature_df(df):
    df_user = pd.DataFrame()
    df_user['id'] = df.index
    df_user.set_index('id',inplace=True)

    start = time()
    df_user['reg_age'] = df['user_created_at'].apply(lambda x: account_age(x))
    dur = time() - start
    print('reg_age:', dur)

    start = time()
    df_user['status_cnt'] = df['user_statuses_count']
    dur = time() - start
    print('statuses_count:', dur)

    start = time()
    df_user['followers_cnt'] = df['user_followers_count']
    dur = time() - start
    print('followers_count:', dur)

    start = time()
    df_user['friends_cnt'] = df['user_friends_count']
    dur = time() - start
    print('friends_count:', dur)

    start = time()
    df_user['verified'] = df['user_verified']
    dur = time() - start
    print('verified:', dur)

    start = time()
    df_user['has_desc'] = df['user_description'].apply(lambda x: x != None)
    dur = time() - start
    print('has_desc:', dur)

    start = time()
    df_user['has_url'] = df['user_url'].apply(lambda x: x != None)
    dur = time() - start
    print('has_url:', dur)

    return df_user

# start = time()
df = tweet_json_to_df(dataset_path)

# build feature table for different feature categories
msg_feat_df = msg_feature_df(df)
usr_feat_df = usr_feature_df(df)

retweet_cnt = retweet_cnt(df[['id_str']])
print(len(msg_feat_df),len(usr_feat_df),len(retweet_cnt))

print('\nValue Frequencies:')
print(retweet_cnt['retweet_count'].value_counts())

df = pd.concat([msg_feat_df, usr_feat_df], axis=1)
df = pd.concat([df, retweet_cnt], axis=1)

df.to_csv('feature_sets\\'+file_name+'_feature_set.csv',sep=';',index = True)
print('\nSaved to ...\\'+file_name+'_feature_set.csv')
