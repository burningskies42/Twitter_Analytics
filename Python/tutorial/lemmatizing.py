# lemmatizing - similiar to stamming. returns synonym
# important to pass the partOfSpeech to 'pos', default is noun
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize('better',pos='a'))
