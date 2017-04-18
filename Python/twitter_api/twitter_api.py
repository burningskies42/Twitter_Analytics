from tweepy import Stream, OAuthHandler, StreamListener
import pandas as pd
import time
import datetime as dt
import json
from os import getcwd, system
import sys

non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)

# save_name.to_pickle('twitter_auth_key.pickle')
api_key = pd.read_pickle('twitter_auth_key.pickle')
system('CLS')
# Greeting
print('''
    ***********************************************************************
    *                                                                     *
    *           Interface and Query laucher for the Twitter API           *
    *                              Version 0.1                            *
    *             Author: Leon Edelmann        Copyright 2017 (c)         *
    *                                                                     *
    ***********************************************************************
''')

# Define stream search parameters
search_term = input("please give search term: ")
search_duration = input("please give search duration in seconds: ")

# listen to data stream from twitter
class listener(StreamListener):
    def __init__(self, api=None):
        super(listener,self).__init__()
        self.start_time = time.time()
        self.count = 0
        self.tw_df = pd.DataFrame()
        self.json_name = search_term + "_dataset.json"
        self.pickle_name = search_term + '_dataset.pickle'

    def on_data(self, data):

        # Time runs out, drop dataframe to file
        if time.time() - self.start_time >= int(search_duration):
            print('\nSearch timed out at ',self.time_now() ,'.' ,str(self.count) ,'tweets collected.')
            dest = str(getcwd())
            print('Data saved to ' + dest + "\\" + self.pickle_name )

            # save to file
            self.tw_df.to_pickle(self.pickle_name)
            return False

        # Time remaining, continue listening on stream
        else:
            try:

                # Defines save file name + converts tweets to dataframe
                # self.save_name = search_term

                data_json = json.loads(data,)
                data_json = pd.Series(data_json)

                if str(data_json['text'].translate(non_bmp_map)).find('RT',0,4) == -1:
                    # Counts tweets collected
                    self.count += 1

                    try:
                        print(json.dumps(data_json['text'],ensure_ascii=False))
                    except Exception as e:
                        print('cant print string',e)

                    self.tw_df = self.tw_df.append(data_json,ignore_index=True)

                    saveFile = open(self.json_name,"a")
                    saveFile.write(data)
                    saveFile.close()

                else:
                    print('------------- retweet, not saved.')

            except Exception as e:
                print("failed on_data,",save_name(e))
                return True  # Don't kill the stream
                print('Stream restarted')

            return(True)


    def on_error(self, status):
        print(sys.stderr, 'Encountered error with status:', status)
        return True  # Don't kill the stream
        print('Stream restarted')

    def on_timeout(self):
        print(sys.stderr, 'Timeout...')
        return True  # Don't kill the stream
        print('Stream restarted')



    def time_now(self):
        return dt.datetime.now().strftime('%H:%M:%S %d/%m/%y')



auth = OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
auth.set_access_token(api_key['access_token'],api_key['access_secret'])

def start_stream(search):
    while True:
        try:
            sapi = Stream(auth, listener())
            sapi.filter(track=[search],languages=['en'])
        except:
            continue

start_stream(search=search_term)

# twitterStream = Stream(auth, listener())
# twitterStream.filter(track=[search_term],languages=['en'])


