from stream_filter.twitter_filter import *
from os import listdir
import tweepy

auth = get_auth()

print('connecting...')

search_duration = 10
search_term = 'Amazon'
ignore_terms = ['gift', 'giftcard', 'giveaway']
count_cap = None

api = tweepy.API(auth)

class Trends():
   def __init__(self):
      self.places_df = pd.DataFrame()
      self.trends_df = pd.DataFrame()
      self.get_trends()

   def isEnglish(self,s):
      try:
         s.encode('ascii')
      except Exception as e:
         return False
      else:
         return True

   def get_trends(self):
      places = api.trends_available()
      for place in places:
         trend = pd.Series(place)
         self.places_df = self.places_df.append(trend, ignore_index=True)

      with open('trends/places.pickle','wb') as fid:
         pickle.dump(self.places_df,fid)
         fid.close()

         self.places_df['woeid'] = self.places_df['woeid'].apply(lambda x:int(x))
         self.places_df = self.places_df[self.places_df['name'] == 'Worldwide']

      for i,row in self.places_df.iterrows():
         trends_in_place = api.trends_place(row['woeid'])[0]

         for trend in trends_in_place['trends']:
            trend['place'] = row['name']
            if self.isEnglish(trend['name']):
               self.trends_df = self.trends_df.append(trend,ignore_index=True)

      # Save to file
      with open('trends/trends.pickle','wb') as fid:
         # trends_in_place = pd.Series(trends_in_place)
         pickle.dump(trends_in_place,fid)
         fid.close()
      self.trends_df.to_csv('trends/trends.csv',sep=';')

t = Trends()
trends = t.trends_df


# quit()
search_term = ['Froome','SUHO','Amazon']

listen = listener(search_duration=search_duration, search_term=search_term, ignore_terms=ignore_terms,save_as='test')
sapi = Stream(auth, listen)
sapi.filter(track=search_term, languages=['en'])