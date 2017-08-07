import WAF_SVM_kernels as waf
import time
import easygui

pth = easygui.fileopenbox()
#            0    1    2    3    4    5    6    7    8    9
num_suff = ['th','st','nd','rd','th','th','th','th','th','th']
i=1

total_run_time = time.time()
while i <=20:
   start_time = time.time()
   cnt = i%5
   print(str(i)+num_suff[cnt],'Itterance')

   if i==1:
      word_classifier = waf.WordsClassifier(gamma=(10**cnt),c_par=1000,load_train='train', pth=pth, from_server=False)
   else:
      word_classifier = waf.WordsClassifier(gamma=(10**cnt),c_par=1000,load_train='train', pth=pth, from_server=False)

   dur = time.strftime('%H:%M:%S',time.gmtime(time.time() - start_time))
   print('Finished',str(i)+" rounds. Round duration:", str(dur))
   i += 1



print('Total run-time',str(time.strftime('%H:%M:%S',time.gmtime(time.time() - total_run_time))))