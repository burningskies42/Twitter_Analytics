from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize,word_tokenize

sample_text = gutenberg.raw('bible-kjv.txt')

tok = sent_tokenize(sample_text)

for sentence in tok[:15]:
    print(sentence)