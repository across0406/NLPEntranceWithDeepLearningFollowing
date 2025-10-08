from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.tokenize import word_tokenize

# stemmer = PorterStemmer()
# If you use Lancaster Stemmer, release commentization the following:
stemmer = LancasterStemmer()

def stem(tokenized: list):
    stemmed_words = list()

    for token in tokenized:
        stemmed_words.append(stemmer.stem(token))

    return stemmed_words

target = "Don't be fooled by the dark sounding name, Mr.Jone's Orphanage is as cheery as cheery goes for a pastry shop."
tokenized = word_tokenize(target)

print('Before stemming:', tokenized)
print('After stemming:', stem(tokenized))
