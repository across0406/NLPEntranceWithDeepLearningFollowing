# Word Tokenize
from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import TreebankWordTokenizer

# Sentence Tokenize
from nltk.tokenize import sent_tokenize

# For Korean language sentence tokenization
import kss

# POS => Part Of Speech

# POS Tagging
from nltk.tag import pos_tag

# For Korean language POS Tagging
from konlpy.tag import Okt
from konlpy.tag import Kkma

import tensorflow as tf


def test_word_tokenization() -> None:
    target = "Don't be fooled by the dark sounding name, Mar.Jone's Orphanage is as cheery as cheery goes for a pastry shop."
    tokenized = word_tokenize(target)
    print('Word Tokenization 1 (with word_tokenize on NLTK):', tokenized)
    tokenized = WordPunctTokenizer().tokenize(target)
    print('Word Tokenization 2 (with word punc tokenizer on NLTK):', tokenized)

    texts = [target]
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_indices = tokenizer.word_index
    word_indices_reversed = {index: word for word, index in word_indices.items()}
    tokenized = [word_indices_reversed[i] for i in sequences[0]]
    print('Word Tokenization 3 (with text tokenizer on keras):', tokenized)

    tree_tokenizer = TreebankWordTokenizer()
    tokenized = tree_tokenizer.tokenize(target)
    print('Word Tokenization 4 (with tree bank word tokenizer on NLTK):',  )


def test_sentence_tokenization() -> None:
    target = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
    print('Sentence Tokenization 1:', sent_tokenize(target))
    target = "I am actively looking for Ph.D. students. and you are a Ph.D student."
    print('Sentence Tokenization 2:', sent_tokenize(target))
    target = '비전과 자연어 처리는 재밌어요. 하지만 표준적인 영어보다 한국어인 경우에는 고려해야할 것이 너무 많아 까다로워요. 이미 한국 사람이면 그걸 본능적으로 느낄걸요?'
    tokenized = kss.split_sentences(target, backend='Mecab') # or kss.split_sentences(target)
    print('Sentence Tokenization 3 (Korean with KSS):', tokenized)


def test_pos_tag() -> None:
    target = "I graduated M.Eng. in CSE, but I want to pursue Ph.D. in CSE."
    tokenized = word_tokenize(target)
    print('Word Tokenization (with work_tokenize on NLTK):', tokenized)
    print('POS Tagging (with pos_tag on NLTK):', pos_tag(tokenized))

    target_kor = '평일에 열심히 일하셨을거에요. 주말입니다. 이제 쉬죠.'
    okt = Okt()
    kkma = Kkma()
    print('OKT Morpheme Analysis:', okt.morphs(target_kor))
    print('OKT POS Tagging:',okt.pos(target_kor))
    print('OKT Noun Extraction:', okt.nouns(target_kor))
    print('KKMA Morpheme Analysis:', kkma.morphs(target_kor))
    print('KKMA POS Tagging:',kkma.pos(target_kor))
    print('KKMA Noun Extraction:', kkma.nouns(target_kor))


if __name__ == '__main__':
    # test_word_tokenization()
    # test_sentence_tokenization()
    test_pos_tag()
