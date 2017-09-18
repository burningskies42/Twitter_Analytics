import words_as_features
import time
import easygui
from os import getcwd,listdir
import pandas as pd
import multiprocessing

# Check False for normal bag-of-words and True for N-Grams
is_ngrams = True

# pth = easygui.fileopenbox()

pth = getcwd() + '/labels'
files = []

for i in listdir(pth):
   if i.find('weighted_dataset') > -1 :
      files.append(pth+'/'+i)

files = sorted(files,key = len)
# files.reverse()

for i in files:
   print(i)

# A matrix to show the classification configs completed:
progress_matrix = pd.DataFrame(columns=[100, 500, 1000, 3000, 5000])
for file in files:
   file = file.split('/')[len(file.split('/')) - 1].split('.')[0]
   se = pd.Series({100: 0, 500: 0, 1000: 0, 3000: 0, 5000: 0})
   se.name = file
   se = se.astype(int)
   progress_matrix = progress_matrix.append(se)

# print(progress_matrix, '\n')
total_run_time = time.time()

if __name__ == '__main__':
   for pth in files:

      new_simulation = words_as_features.WordsClassifier()
      new_simulation.fetch_tweets(pth=pth
                                  , remove_stopwords=(not is_ngrams)
                                  , ngrams=is_ngrams, n_min=2, n_max=5
                                  , with_print=False)

      for feature_cnt in [5000,3000,1000,500,100]:
      # for feature_cnt in [100]:

         new_simulation.build_features(num_features=feature_cnt)
         new_simulation.train_test_split(with_print=True)

         # new_simulation.train(num_features=,with_trees=False)

         processes = []
         i = 1
         while i <= 10:
            # Args: feature_cnt,with_trees,ngrams,n_min,n_max,with_print
            # p = multiprocessing.Process(target=new_simulation.train,args=[False,is_ngrams, 2, 5,True])
            new_simulation.train(False,is_ngrams, 2, 5,True)
            # processes.append(p)
            i+=1
         # words_as_features.classification_simulation(pth=pth,feature_cnt=feature_cnt)
         #
         # for pr in processes:
         #    pr.start()


         # for pr in processes:
         #    pr.join()
            fl = pth.split('/')[len(pth.split('/')) - 1].split('.')[0]
            progress_matrix[feature_cnt][fl] += 1

            print(progress_matrix, '\n')


print('Total run-time',str(time.strftime('%H:%M:%S',time.gmtime(time.time() - total_run_time))))