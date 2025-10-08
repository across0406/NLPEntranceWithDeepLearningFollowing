from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


def penn_to_wn(tag: str):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('V'):
        return wn.VERB
    elif tag.startswith('R'):
        return wn.ADV
    else:
        return None


def lemmatize(pos_tagged_words: list):
    lemmatized_words = list()

    for word, tag in pos_tagged_words:
        wn_tag = penn_to_wn(tag)
        if wn_tag in (wn.NOUN, wn.ADJ, wn.ADV, wn.VERB):
            lemmatized_words.append(lemmatizer.lemmatize(word, wn_tag))
        else:
            lemmatized_words.append(word)

    return lemmatized_words

lemmatizer = WordNetLemmatizer()
target = "Don't be fooled by the dark sounding name, Mr.Jone's Orphanage is as cheery as cheery goes for a pastry shop."
tokenized = word_tokenize(target)
tagged = pos_tag(tokenized)
print('POS Tagging (with pos_tag on NLTK):', tagged)
print('Before lemmatization:', tokenized)
print('After lemmatization:', lemmatize(tagged))
