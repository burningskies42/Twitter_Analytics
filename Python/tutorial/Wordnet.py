from nltk.corpus import wordnet

# syns = wordnet.synsets('program')

# print words
# for word in syns:
#     print(word.lemmas())

# # defitions and examples
# for word in syns:
#     print(word.name(),':',word.definition(),'\n',word.examples(),'\n')

# # Synonyms and antonyms
# synonyms = []
# antonyms = []
#
# for syn in wordnet.synsets("good"):
#     for l in syn.lemmas():
#         # print('l:',l)
#         synonyms.append(l.name())
#         if l.antonyms():
#             antonyms.append(l.antonyms()[0].name())
#
# # print(set(synonyms))
# # print(set(antonyms))

# Semantic similiarity
w1 = wordnet.synset('boat.n.01')
w2 = wordnet.synset('ship.n.01')
w3 = wordnet.synset('car.n.01')
w4 = wordnet.synset('cat.n.01')

# Wu and Palmer similarity
print(w2.wup_similarity(w3))