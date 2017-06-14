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


import stream_filter.filtergui
from sys import argv
from PyQt4 import QtGui, QtCore

from feature_tk.features import tweets_to_featureset

# listen to data stream from twitter
class listener(StreamListener):
    def __init__(self, search_duration, search_term, ignore_terms, api=None):
        super(listener,self).__init__()
        self.start_time = time.time()
        self.count = 0
        self.tw_df = pd.DataFrame()
        self.json_name = path.join('captured_tweets',search_term + "_dataset.json")
        self.pickle_name = path.join('captured_tweets',search_term + "_dataset.pickle")
        self.search_duration = search_duration
        self.ignore_terms = ignore_terms
        self.recorded_ids = set()

    def on_connect(self):
        print('Connected to server ...\n')

    def on_data(self, data):

        retweeted = False
        quoted = False
        # Time runs out, drop dataframe to file
        if time.time() - self.start_time >= float(self.search_duration):
            print('\nSearch timed out at ',self.time_now() ,'.' ,str(self.count) ,'tweets collected.')
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

                if retweeted:
                    twt_text = '------------- retweet source: ' + twt_text
                elif quoted:
                    twt_text = '------------- quote source: ' + twt_text
                print(twt_text)

                # Check if already captured, if yes ignore
                if data_json['id'] not in self.recorded_ids:
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

    search_duration = input("please give search duration in seconds: ")

    # Connects to twitter API
    auth = get_auth()

    while True:
        # try:
        print('connecting...')
        listen = listener(search_duration=search_duration,search_term=search_term,ignore_terms=ignore_terms)
        sapi = Stream(auth,listen)
        sapi.filter(track=[search_term],languages=['en'])
        # except Exception as e:
        #     print('error:',e)
        #     continue

        if listen.start_time + float(search_duration) +2  > time.time() :
            print('breaking loop')
            quit()


# def start_stream_pyqt(ignore_items=['gift','giftcard','giveaway']):
#     app = QtGui.QApplication(argv)
#     GUI = stream_filter.filtergui.Window(ignore_items)
#     GUI.setWindowState(GUI.windowState() & ~QtCore.Qt.WindowMinimized | QtCore.Qt.WindowActive)
#     GUI.activateWindow()
#     GUI.exec()
#
#
#     # Define stream search parameters
#     # search_term = input("please give search term: ")
#     # ignore_terms = input("ignore tweets containing (optional, semicolon-separated): ")
#
#     # nead to rethink this entire scheme for the search GUI
#     #
#     #
#
#     # search_term = GUI.sea
#     ignore_terms = GUI.ignore_terms
#
#     if len(ignore_terms) == 0:
#         ignore_terms = ['gift', 'giftcard', 'giveaway']
#     else:
#         ignore_terms = [w.replace(' ', '') for w in ignore_terms.split(';') if len(w.replace(' ', '')) > 0]
#     print(ignore_terms)
#
#     search_duration = input("please give search duration in seconds: ")
#
#     # Connects to twitter API
#     auth = get_auth()
#
#     while True:
#         try:
#             print('connecting...')
#             listen = listener(search_duration=search_duration, search_term=search_term, ignore_terms=ignore_terms)
#             sapi = Stream(auth, listen)
#             sapi.filter(track=[search_term], languages=['en'])
#         except Exception as e:
#             print('error:', e)
#             continue
#
#         if listen.start_time + float(search_duration) + 2 > time.time():
#             print('breaking loop')
#             quit()



