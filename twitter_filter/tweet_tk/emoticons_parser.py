from json import dumps
from re import compile,VERBOSE
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import Request,urlopen
from pathlib import Path

# downloads emoticons sentiment scores from web
def emoji_sentiment_ranking_dict():
    # download emoji sentiment ranking from website
    url = 'http://kt.ijs.si/data/Emoji_sentiment_ranking/'
    resp = urlopen(Request(url))
    page = resp.read()

    # covert the table to a dataframe
    soup = BeautifulSoup(page,'lxml')
    table = soup.find("table")
    data = []
    table_body = table.find('tbody')

    rows = table_body.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        data.append([ele for ele in cols if ele]) # Get rid of empty values

    df = pd.DataFrame(data, columns=['char','image','unicode','occurrences',
                                      'position','neg','neut','pos','sentiment',
                                      'uni_name','uni_block'])
    # convet df to dict of form : {utf8_emot:sentiment_value}
    df = df[['char','sentiment']]
    df.set_index('char',inplace=True)
    df = df['sentiment'].to_dict()
    pd.to_pickle(df,'Sentiment_Scores.pickle')
    return df

# regex pattern to find emoticons
emoticon_pattern=compile(r" " " [\U0001F600-\U0001F64F] # emoticons \
                                 |\
                                 [\U0001F300-\U0001F5FF] # symbols & pictographs\
                                 |\
                                 [\U0001F680-\U0001F6FF] # transport & map symbols\
                                 |\
                                 [\U0001F1E0-\U0001F1FF] # flags (iOS)\
                          " " ", VERBOSE)

# import sentiment scores
if Path('Sentiment_Scores.pickle').is_file():
    emoticons_sentiment = pd.read_pickle('Sentiment_Scores.pickle')
    # print('read sentiment score from pickle')
else:
    emoticons_sentiment = emoji_sentiment_ranking_dict()
    print('downloaded sentiment score from pickle')

def emoticons_score(text):
    emoticons = emoticon_pattern.findall(text)
    em_list = []
    score = 0
    for emoticon in emoticons:
        emoticon = dumps(emoticon,ensure_ascii=False).strip('"')
        if emoticon in emoticons_sentiment.keys():
            em_list.append(emoticon)
            score += float(emoticons_sentiment[emoticon])
    if len(em_list) > 0:
        score /= len(em_list)

    return score

def surr_pair_to_utf(str):
    return str.strip('"').encode('utf-16', 'surrogatepass').decode('utf-16')


