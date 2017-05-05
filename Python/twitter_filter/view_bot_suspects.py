import pickle

suspects_file = open('bot_suspects\\bot_suspects.pickle','rb')
suspects = pickle.load(suspects_file)
i=0

for s in suspects:
    i+=1
    print(str(i)+'.',s)