from tweepy import Stream, OAuthHandler, StreamListener
from os import getcwd, system, path
import pandas as pd
import datetime as dt
import numpy as np
import time
import json
import sys
import pickle
from sklearn import preprocessing

# ----------------------------------------------------------
#
#    Dont start directly ! use stream.py instead !
#
# ----------------------------------------------------------

# listen to data stream from twitter
class listener(StreamListener):
    def __init__(self, search_duration, search_term, ignore_terms,tweet_cnt_limit = -1, api=None):
        super(listener,self).__init__()
        self.start_time = time.time()
        self.count = 0
        self.tw_df = pd.DataFrame()
        self.json_name = path.join('captured_tweets',search_term + "_dataset.json")
        self.pickle_name = path.join('captured_tweets',search_term + "_dataset.pickle")
        self.search_duration = search_duration
        self.ignore_terms = ignore_terms
        self.tweet_cnt_limit = tweet_cnt_limit
        self.recorded_ids = set()

        if self.tweet_cnt_limit == -1:
            self.count_cap = False
        else:
            self.count_cap = True


    def on_connect(self):
        print('Connected to server ...\n')

    def on_data(self, data):
        retweeted = False
        quoted = False

        # Time runs out, drop dataframe to file
        if time.time() - self.start_time >= float(self.search_duration) :
            print('\nSearch timed out at ',self.time_now() ,'.' ,str(self.count) ,'tweets collected.')
            self.save_tweets()
            return False

        # Tweet cap reached, drop dataframe to file
        if self.count_cap and int(self.count)>=int(self.tweet_cnt_limit) :
            print('\nSearch reached tweet count limit at ',self.time_now() ,'.' ,str(self.count) ,'tweets collected.')
            self.save_tweets()
            return False

        # Time remaining, continue listening on stream
        else:
            # Defines save file name + converts tweets to dataframe
            data_json = json.loads(data)
            data_json = pd.Series(data_json)

            try:
                text = str(data_json['text'])
            except Exception as e:
                print(data_json+'\n\n'+'error reading tweet, skippnig ...')
                quit()
            # return(True)

            # check if retweet
            # Disregard retweets
            if text.find('RT', 0, 4) != -1:
                retweeted = True
                if 'retweeted_status' in data_json.keys():

                    # extract the original Tweet and record it instead of the current tweet
                    data_json = data_json['retweeted_status']
                    data = json.dumps(data_json)+'\r\n'

                else:
                    print('no retweeted_status:' + str(data_json['id']))
                    # quit()
            elif 'quoted_status' in data_json.keys():
                quoted = True
                # extract the original Tweet and record it instead of the current tweet
                data_json = data_json['quoted_status']
                data = json.dumps(data_json) + '\r\n'


            # Check for ignore terms
            if not any(ignore_str in text.lower() for ignore_str in self.ignore_terms):
                # Output tweets captured and amount, ensure_ascii prevents emoticons
                twt_text = json.dumps(data_json['text'],ensure_ascii=False)

                # Check if already captured, if yes ignore
                if data_json['id'] not in self.recorded_ids:

                    if retweeted:
                        twt_text = '------------- retweet source: ' + twt_text
                    elif quoted:
                        twt_text = '------------- quote source: ' + twt_text
                    print(str(self.count+1)+'.',twt_text)


                    # appends tweet to json (backup for failure on pickle)
                    self.recorded_ids.add(data_json['id'])
                    self.count += 1

                    # write pickle
                    self.tw_df = self.tw_df.append(data_json, ignore_index=True)

                    # write json
                    saveFile = open(self.json_name,"a")
                    saveFile.write(data)
                    saveFile.close()

                else:
                    print('------------- already captured. ignoring',str(data_json['id']))

            # Disregard tweets with ignore-terms
            else:
                print('------------- contains ignore-term, not saved.')


            return(True)

    def on_error(self, status):
        # Timeout error
        if status == 420:
            print(sys.stderr,'Stream restarted after time-out error 420')
            return True

        # Other error types
        else:
            print(sys.stderr,'Encountered error with status:', status)
            print(sys.stderr,'Stream restarted after error')
            return True  # Don't kill the stream

    def on_timeout(self):
        print(sys.stderr, 'Timeout...')
        print('Stream restarted after time-out')
        return True     # Don't kill the stream

    def time_now(self):
        return dt.datetime.now().strftime('%H:%M:%S %d/%m/%y')

    # save to pickle
    def save_tweets(self):
        dest = str(getcwd())
        self.tw_df.to_pickle(self.pickle_name)
        print('Data saved to '+dest+"\\"+self.pickle_name)

def get_auth():
    api_key = pd.read_pickle('tweet_tk\\auth\\twitter_auth_key.pickle')
    auth = OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
    auth.set_access_token(api_key['access_token'], api_key['access_secret'])
    return auth

# Captures tweet according to a search term
def start_stream():

    system('CLS')
    # Greeting
    print('''
                    ***********************************************************************
                    *                                                                     *
                    *           Interface and Query laucher for the Twitter API           *
                    *                              Version 0.2                            *
                    *             Author: Leon Edelmann        Copyright 2017 (c)         *
                    *                                                                     *
                    ***********************************************************************
    ''')

    # Define stream search parameters
    search_term = input("please give search term: ")
    ignore_terms = input("ignore tweets containing (optional, semicolon-separated): ")
    if len(ignore_terms) == 0:
        ignore_terms = ['gift', 'giftcard', 'giveaway']
    else:
        ignore_terms = [w.replace(' ', '') for w in ignore_terms.split(';') if len(w.replace(' ', '')) > 0]
    print(ignore_terms)

    incorrect_input = True
    while incorrect_input:
        search_duration = input("please give search duration in seconds: ")
        if len(search_duration) == 0:
            search_duration = 3600
            incorrect_input = False
        elif str(search_duration).isnumeric():
            incorrect_input = False

    incorrect_input = True
    while incorrect_input:
        count_cap = input("please give max # of tweets to be collected (leave blank - no cap): ")
        if len(count_cap) == 0:
            count_cap = -1
            incorrect_input = False
        elif count_cap.isnumeric():
            incorrect_input = False
        else:
            print('non numeric input, leave blank for no cap')

    # Connects to twitter API
    auth = get_auth()

    while True:
        print('connecting...')
        listen = listener(search_duration=search_duration,search_term=search_term,ignore_terms=ignore_terms,tweet_cnt_limit=count_cap)
        sapi = Stream(auth,listen)
        sapi.filter(track=[search_term],languages=['en'])

        if (listen.start_time + float(search_duration) +2  > time.time()) or (listen.count_cap and listen.count == listen.tweet_cnt_limit) :
            print('breaking loop')
            quit()


