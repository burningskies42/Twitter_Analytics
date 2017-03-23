from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time

consumer_key = '0TgapeAfCQOnBdY1Nx1YycjPK'
consumer_secret = 't1JvqhBKX5b6xO1HeSBYKi8kLBg3tfrqhouIqnTZ0eTgu1ZJ44'
access_token = '2572138643-afD4rDaMr1QTuz5JvUZYsBUxVRulOaQJSG5RWdG'
access_secret = 'hzcrINwNViYdBhN5rOBoxaTsphhq9aS8r6rCPkZm85LNr'

search_term = input("please give search term:")

class listener(StreamListener):

    def on_data(self, data):

        try:
            print(data)
            str = search_term + "_db.csv"
            tweet = data.split(',"text":"')[1].split(',"source":')[0]
            #print(tweet)

            saveFile = open(str,"a")
            saveFile.write(data)
            saveFile.close()
            return True
        except Exception as e:
            print("failed ondata,",str(e))

        return(True)


    def on_error(self, status):
        print(status)


auth = OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=[search_term])

