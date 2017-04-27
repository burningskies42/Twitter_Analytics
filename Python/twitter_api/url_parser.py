import bs4 as bs
import urllib.request
import pandas as pd

def most_pop_urls():
    source = urllib.request.urlopen('https://en.wikipedia.org/wiki/List_of_most_popular_websites').read()
    soup = bs.BeautifulSoup(source,'lxml')
    table = soup.table

    table_rows = table.find_all('tr')

    rating_table = pd.DataFrame()
    # columns=('Site','Domain','Alexa','SimilarWeb','Type','Country')

    for tr in table_rows:
        td = tr.find_all('td')
        row = [i.text for i in td]
        if len(row)>0:
            # print(len(row))
            rating_table = rating_table.append(pd.Series(row),ignore_index=True)

    rating_table.columns = ['Site','Domain','Alexa','SimilarWeb','Type','Country']
    return rating_table

