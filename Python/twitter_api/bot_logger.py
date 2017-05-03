import pickle

with open('bot_suspects//bot_suspects.txt','rb') as fp:
    itemlist = pickle.load(fp)

print(itemlist)