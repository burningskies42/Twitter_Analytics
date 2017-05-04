from tweet_tk.bots import *

suspects_file = open('bot_suspects\\bot_suspects.pickle','rb')

suspects = load(suspects_file)
print(suspects)