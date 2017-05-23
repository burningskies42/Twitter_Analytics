from pickle import dump,load

def add_suspects(set_of_new_ids):
    suspect_file = open('bot_suspects\\bot_suspects.pickle','rb')
    old_suspects = set(load(suspect_file))
    print('------------------\n'+str(len(old_suspects))
          ,'old suspects in list')
    suspect_file.close()

    unique_suspects = set_of_new_ids.difference(old_suspects)
    print(len(set_of_new_ids),'bot suspects in current dataset, of them',
          len(unique_suspects), 'are new')

    updated_suspects = set(old_suspects).union(unique_suspects)
    # updated_suspects = set_of_new_ids
    print('updated suspect list length is',
          len(updated_suspects),
          '\n------------------')
    suspect_file = open('bot_suspects\\bot_suspects.pickle', 'wb')
    dump(updated_suspects, suspect_file)
    suspect_file.close()

def is_suspect(user_id):
   suspect_file = open('bot_suspects\\bot_suspects.pickle', 'rb')
   suspects = set(load(suspect_file))
   suspect_file.close()

   return user_id in suspects

