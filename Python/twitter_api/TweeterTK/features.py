from re import sub,MULTILINE
from difflib import SequenceMatcher
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime,timezone
from textblob import Blobber
from textblob_aptagger import PerceptronTagger

tb = Blobber(pos_tagger=PerceptronTagger())

import bs4 as bs
import urllib.request
from pandas import DataFrame,Series

def most_pop_urls():
    try:
        source = urllib.request.urlopen('https://en.wikipedia.org/wiki/List_of_most_popular_websites').read()
        soup = bs.BeautifulSoup(source,'lxml')
        table = soup.table

        table_rows = table.find_all('tr')

        rating_table = DataFrame()
        # columns=('Site','Domain','Alexa','SimilarWeb','Type','Country')

        for tr in table_rows:
            td = tr.find_all('td')
            row = [i.text for i in td]
            if len(row)>0:
                # print(len(row))
                rating_table = rating_table.append(Series(row),ignore_index=True)

        rating_table.columns = ['Site','Domain','Alexa','SimilarWeb','Type','Country']
        return rating_table
    except Exception as e:
        print(e)

# generate list of most popular websites
most_pop_urls = list(most_pop_urls()['Domain'])
print('downloaded most popular domains\n')

def similar(df):
    sent_df = df.to_frame()
    sent_df['similiarity'] = ''

    for i,sentence in sent_df.iterrows():
        max_ratio = 0
        for j,another in sent_df.iterrows():

            # if different tweets, compare to max
            if i != j:
                simil= SequenceMatcher(None, sentence, another).ratio()
                if simil == 1:
                    sent_df['similiarity'][i] = 1
                    # sent_df['similiarity'][j] = 1
                    break

                elif simil > max_ratio:
                    max_ratio = simil

        sent_df['similiarity'][i] = max_ratio
    return sent_df['similiarity']

def tokenize_and_filter(sentence):

    sentence = word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    words = []
    for w in sentence:
        if w.lower() not in stop_words:
                words.append(w)
    # words = [w for w in sentence if w.lower() not in stop_words]
    tagged = pos_tag(words)
    return tagged

def has_pronoun(text):
    tagged_words = tb(text)
    dc = [x[1] for x in tagged_words.pos_tags]
    return ('PRP' in dc)

def count_upper(text):
    return sum(1 for c in text if c.isupper())/len(text)

def get_urls(entity):
    if len(entity) > 0:
        urls = [x['expanded_url'] for x in entity['urls']]

        flag = False
        url = None
        for each in urls:
            url = each.split('/')[2]
            if len(url.split('www.')) > 1:
                url = url.split('www.')[1]

            if url in most_pop_urls:
                flag = True

    return flag

def clear_urls(text):
    clear_text = sub(r'https?:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text,flags=MULTILINE)
    return clear_text

def account_age(user_created_at):
    creataion_date = datetime.strptime(user_created_at, '%a %b %d %H:%M:%S %z %Y')
    return (datetime.now(timezone.utc) - creataion_date).days