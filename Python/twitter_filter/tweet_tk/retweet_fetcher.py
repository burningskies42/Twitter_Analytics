from pandas import DataFrame,read_pickle
from tweepy import OAuthHandler, API
from os import getcwd

# connect to twitter server
def retweet_cnt(id_list):
    api_key = read_pickle(getcwd()+'\\tweet_tk\\auth\\twitter_auth_key.pickle')
    auth = OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
    auth.set_access_token(api_key['access_token'],api_key['access_secret'])
    api = API(auth)
    print('\nQuerying tweet ids from server:')

    tweets_df = id_list
    i=0
    tweets_df = DataFrame(columns=['retweet_count'])

    while i <= len(id_list):
        # query to get list of STATUS objects from ids

        ids_chunk = id_list[i:min(i + 100, len(id_list))]

        print('queried',i,'to',min(i + 100, len(id_list)))
        # print(list(ids_chunk))
        tweets_chunk = API.statuses_lookup(api, list(ids_chunk.index))
        # tweets_chunk = tweets_chunk['retweet_count']
        tweets_dict = {}
        for tweet in tweets_chunk:
            tweets_dict[tweet._json['id']] =  tweet._json['retweet_count']

        tweets_dict = DataFrame.from_dict(tweets_dict,orient='index')
        tweets_dict.columns = ['retweet_count']
        tweets_df = tweets_df.append(tweets_dict)

        i += 100

    # tweets_df.set_index('id',inplace=True)
    return tweets_df

