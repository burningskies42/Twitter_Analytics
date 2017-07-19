import pandas as pd
import pickle
import re

df = pd.read_csv('C:/Twitter_Analytics/Python/twitter_filter/classifiers/words_as_features/latest_dataset.csv',sep=';',encoding = "ISO-8859-1")

words = {}

for ind,row in df.iterrows():
   sentence = [w.replace("'","").replace(' ','') for w in (row['text'][1:-1]).split(',')]
   for w in sentence:
      if w in words.keys():
         words[w]+=1
      else:
         words[w]=1

top50 = sorted(words.items(), key=lambda x:x[1])[:50]

top50 = [w[0] for w in top50]

top_words_labels={}
for w in top50:
   top_words_labels[w] = {'news':0,'spam':0}

for ind,row in df.iterrows():
   for w in top_words_labels.keys():
      if row['text'].find(w) != -1:
         if row['label'] == 'news':
            top_words_labels[w]['news']+=1
         else:
            top_words_labels[w]['spam'] += 1

df = pd.DataFrame(columns=['word','news','spam'])

for word,vals in top_words_labels.items():
   dic = {'word':word,'news':vals['news'],'spam':vals['spam']}
   se = pd.Series(dic)
   df = df.append(se,ignore_index=True)

# df['ratio'] = df['news']/df['spam']
df.sort_values(by='ratio',inplace=True,ascending=False)
print(df)

