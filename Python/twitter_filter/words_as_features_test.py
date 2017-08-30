import words_as_features
import time
import easygui

pth = easygui.fileopenbox()
#            0    1    2    3    4    5    6    7    8    9
num_suff = ['th','st','nd','rd','th','th','th','th','th','th']
i=1

total_run_time = time.time()
while i <=50:
   start_time = time.time()
   print(str(i)+num_suff[(i%9)],'Itterance')

   if i==1:
      # word_classifier = words_as_features.WordsClassifier('train', pth=pth, from_server=True,num_features=3000)
      word_classifier = words_as_features.WordsClassifier('train', pth=pth, from_server=True,num_features=3000)
   else:
      word_classifier = words_as_features.WordsClassifier('train', pth=pth, from_server=False,num_features=3000)

   dur = time.strftime('%H:%M:%S',time.gmtime(time.time() - start_time))
   print('Finished',str(i)+" rounds. Round duration:", str(dur))
   i += 1



print('Total run-time',str(time.strftime('%H:%M:%S',time.gmtime(time.time() - total_run_time))))