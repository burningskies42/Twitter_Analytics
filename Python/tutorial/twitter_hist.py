import tweepy
import pandas as pd

keys = pd.read_pickle('')
consumer_key = '***'
consumer_secret = '***'
access_token = '***'
access_token_secret = '***'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)