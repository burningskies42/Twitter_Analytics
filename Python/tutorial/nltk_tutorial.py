from nltk.tokenize import PunktSentenceTokenizer
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
from nltk.corpus import state_union
import nltk
from matplotlib import *

# # Stop Words
# example_sentence = 'this is an example sentence showing off stop word filtration.'
# def tokenize_and_filter(sentence):
#     sentence = word_tokenize(sentence)
#     stop_words = set(stopwords.words('english'))
#     words = [w for w in sentence if w not in stop_words]
#     return words
#
# print(set(stopwords.words('english')))


# # Stemming - reduce words to their root
# ps = PorterStemmer()
#
# example_words = ["python","pythoner","pythoning","pythoned","pythonly"]
# new_text = "It is important to be very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
#
# # for w in example_words:
# #     print(ps.stem(w))
#
# words = word_tokenize(new_text)
# for w in words:
#     print(ps.stem(w))

# Taging parts of speech
train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            # chunkGram = r'''Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}'''
            chunkGram = r'''Chunk: {<.*>+}
                                        }<VB.?|IN|DT|TO>+{'''

            ChunkParser = nltk.RegexpParser(chunkGram)
            chunked = ChunkParser.parse(tagged)
            chunked.draw()

    except Exception as e:
        print(str(e))

process_content()

