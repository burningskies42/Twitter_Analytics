from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

example_sentence = 'this is an example sentence showing off stop word filtration.'
def tokenize_and_filter(sentence):
    sentence = word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    words = [w for w in sentence if w not in stop_words]
    return words

print(set(stopwords.words('english')))
