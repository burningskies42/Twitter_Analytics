from pandas import DataFrame,read_pickle
from tweepy import OAuthHandler, API
from os import getcwd

# connect to twitter server
def retweet_cnt(id_list,with_timing = False):
    api_key = read_pickle(getcwd()+'\\tweet_tk\\auth\\twitter_auth_key.pickle')
    auth = OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
    auth.set_access_token(api_key['access_token'],api_key['access_secret'])
    api = API(auth)
    if with_timing: print('\nQuerying tweet ids from server:')

    i=0
    tweets_dict = {}

    while i <= len(id_list):

        # query to get list of STATUS objects from ids
        ids_chunk = id_list[i:min(i + 100, len(id_list))]
        if with_timing: print('retweets counted for:',i,'to',min(i + 100, len(id_list)))

        tweets_chunk = API.statuses_lookup(api, ids_chunk)


        for tweet in tweets_chunk:
            tweets_dict[tweet._json['id']] =  tweet._json['retweet_count']


        i += 100

    tweets_df = DataFrame.from_dict(tweets_dict,orient='index')
    print(tweets_df)
    tweets_df.columns = ['retweet_count']

    return tweets_df

