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
        self.save_name = ""

    def on_data(self, data):
        save_name = ""

        # Time runs out, drop dataframe to file
        if time.time() - self.start_time >= int(search_duration):
            print('\nSearch timed out at ',self.time_now() ,'.' ,str(self.count) ,'tweets collected.')
            dest = str(getcwd())
            print('Data saved to' + dest + "\\" + self.save_name )

            # save to file
            self.tw_df.to_json(self.save_name,orient='records',lines=True)

            return False

        # Time remaining, continue listening on stream
        else:
            try:
                # Counts tweets collected
                self.count += 1

                # Defines save file name + converts tweets to dataframe
                self.save_name = search_term + "_db.json"
                data_json = json.loads(data)
                data_json = pd.Series(data_json)

                print(data_json['text'].translate(non_bmp_map))

                self.tw_df = self.tw_df.append(data_json,ignore_index=True)
                # saveFile = open(save_name,"a")
                # saveFile.write(data)
                # saveFile.close()

            except Exception as e:
                print("failed ondata,",save_name(e))

            return(True)


    def on_error(self, status):
        print(status)

    def time_now(self):
        return dt.datetime.now().strftime('%H:%M:%S %d/%m/%y')



auth = OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
auth.set_access_token(api_key['access_token'],api_key['access_secret'])

twitterStream = Stream(auth, listener())
twitterStream.filter(track=[search_term])


