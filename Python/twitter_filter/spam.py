import pickle
from os import getcwd

with open(getcwd() + "\\classifiers\\words_as_features\\all_Words.pickle", "rb") as fid:
   a = pickle.load( fid)
i= 0
for key,val in sorted(a.items(), key=lambda x: x[1], reverse=True):
   if len(key.split(' '))==5:
      print(key,val)
      i+=1

      if i == 20:
         break