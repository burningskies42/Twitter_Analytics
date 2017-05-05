from tweepy import Stream, OAuthHandler, StreamListener
from os import getcwd, system, path
import pandas as pd
import datetime as dt
import time
import json
import sys

api_key = pd.read_pickle('tweet_tk\\auth\\twitter_auth_key.pickle')
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
ignore_terms = [w.replace(' ','') for w in ignore_terms.split(';') if len(w.replace(' ',''))>0]
print(ignore_terms)

search_duration = input("please give search duration in seconds: ")




# listen to data stream from twitter
class listener(StreamListener):
    def __init__(self, api=None):
        super(listener,self).__init__()
        self.start_time = time.time()
        self.count = 0
        self.tw_df = pd.DataFrame()
        self.json_name = path.join('captured_tweets',search_term + "_dataset.json")
        self.pickle_name = path.join('captured_tweets',search_term + "_dataset.pickle")
    def on_connect(self):
        print('Connected to server ...\n')
    def on_data(self, data):

        # Time runs out, drop dataframe to file
        if time.time() - self.start_time >= float(search_duration):
            print('\nSearch timed out at ',self.time_now() ,'.' ,str(self.count) ,'tweets collected.')
            self.save_tweets()
            return False

        # Time remaining, continue listening on stream
        else:
            # Defines save file name + converts tweets to dataframe
            data_json = json.loads(data)
            data_json = pd.Series(data_json)

            # check if retweet
            text = str(data_json['text'])
            if text.find('RT',0,4) == -1:

                # Check for ignore terms
                if not any(ignore_str in text.lower() for ignore_str in ignore_terms):
                    # Output tweets captured and amount, ensure_ascii prevents emoticons
                    self.count += 1
                    twt_text = json.dumps(data_json['text'],ensure_ascii=False)
                    # twt_text.replace('\n')
                    print(twt_text)

                    # appends tweet to json (backup for failure on pickle)
                    self.tw_df = self.tw_df.append(data_json,ignore_index=True)
                    saveFile = open(self.json_name,"a")
                    saveFile.write(data)
                    saveFile.close()

                # Disregard tweets with ignore-terms
                else:
                    print('------------- contains ignore-term, not saved.')

            # Disregard retweets
            else:
                print('------------- retweet, not saved.')

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


# Connects to twitter API
auth = OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
auth.set_access_token(api_key['access_token'],api_key['access_secret'])

# Captures tweet according to a search term
def start_stream(search):
        print('connecting...')
        listen = listener()
        sapi = Stream(auth,listen)
        sapi.filter(track=[search],languages=['en'])

start_stream(search=search_term)


