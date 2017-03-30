from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import pandas as pd
import time
import datetime as dt

# save_name.to_pickle('twitter_auth_key.pickle')
api_key = pd.read_pickle('twitter_auth_key.pickle')

# Define stream search parameters
search_term = input("please give search term: ")
search_duration = input("please give search duration in seconds: ")

# listen to data stream from twitter
class listener(StreamListener):
    def __init__(self, api=None):
        super(listener,self).__init__()
        self.start_time = time.time()
        self.count = 0

    def on_data(self, data):
        if time.time() - self.start_time >= int(search_duration):
            time_now = dt.datetime.now().strftime('%H:%M:%S %d/%m/%y')
            cnt =  str(self.count)
            print('\nSearch timed out at ',time_now ,'.' ,cnt ,'tweets collected.')
            return False
        else:
            try:
                self.count += 1
                # print(self.count)
                save_name = search_term + "_db.csv"

                tweet = data.split(',"text":"')[1].split(',"source":')[0]
                print(tweet)

                # data = data.replace('","','";"')
                saveFile = open(save_name,"a")
                saveFile.write(data)
                saveFile.close()
            except Exception as e:
                print("failed ondata,",save_name(e))

            return(True)


    def on_error(self, status):
        print(status)


auth = OAuthHandler(api_key['consumer_key'], api_key['consumer_secret'])
auth.set_access_token(api_key['access_token'],api_key['access_secret'])

twitterStream = Stream(auth, listener())
twitterStream.filter(track=[search_term])

