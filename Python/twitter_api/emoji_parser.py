import nltk
import json, re
import pandas as pd
from emoji_sentiment import *



emoticons_sentiment = get_Emoji_sentiment_ranking()
emoticons_sentiment.set_index('char',inplace=True)
emoticons_sentiment = emoticons_sentiment['sentiment'].to_dict()
emoticons_sentiment['🙄'] = 0

def surr_pair_to_utf(str):
    return str.strip('"').encode('utf-16', 'surrogatepass').decode('utf-16')
# open dataset
with open('amazon_dataset.json','r') as file:
    lines = [line.rstrip('\n') for line in file]

# remove empty lines
lst = []
for i in range(len(lines)):
    if i%2 == 0:
        lst.append(json.loads(lines[i]))

# emoji_pattern = re.compile('[\U0001F300-\U0001F64F]')

emoji_pattern=re.compile(r" " " [\U0001F600-\U0001F64F] # emoticons \
                                 |\
                                 [\U0001F300-\U0001F5FF] # symbols & pictographs\
                                 |\
                                 [\U0001F680-\U0001F6FF] # transport & map symbols\
                                 |\
                                 [\U0001F1E0-\U0001F1FF] # flags (iOS)\
                          " " ", re.VERBOSE)


# find all emoticons
for dict_input in lst:
    # dict_input = json.loads(json_input)
    text = dict_input['text']
    screen_name = dict_input['user']
    emojis = emoji_pattern.findall(text)

    print(len(emojis), 'chars found in post ',dict_input['id'])
    for emoji in emojis:
        em = json.dumps(emoji,ensure_ascii=False).strip('"')
        print(em,emoticons_sentiment[em])
