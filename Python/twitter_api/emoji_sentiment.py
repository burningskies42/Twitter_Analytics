from pandas import DataFrame
from bs4 import BeautifulSoup
from urllib.request import Request,urlopen

def get_Emoji_sentiment_ranking():
    url = 'http://kt.ijs.si/data/Emoji_sentiment_ranking/'

    req = Request(url)
    resp = urlopen(req)
    page = resp.read()

    soup = BeautifulSoup(page,'lxml')
    table = soup.find("table")
    data = []
    table_body = table.find('tbody')

    rows = table_body.find_all('tr')
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        data.append([ele for ele in cols if ele]) # Get rid of empty values

    df = DataFrame(data, columns=['char','image','unicode','occurrences',
                                      'position','neg','neut','pos','sentiment',
                                      'uni_name','uni_block'])
    return df[['char','sentiment']]
