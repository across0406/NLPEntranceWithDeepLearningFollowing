from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import timeit
from konlpy.tag import Okt


def check_stop_words() -> None:
    stop_word_list = stopwords.words('english')
    print('The number of stop-words:', len(stop_word_list))
    print('10 stop-words print:', stop_word_list[0:10])


def measure_filter_with_stop_words() -> None:
    consumed_time = list()
    filter_setup_set = """
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
target = 'I graduated M.Eng. in CSE, but I want to pursue Ph.D. in CSE.'
word_tokens = word_tokenize(target)

result = list()
stop_words = set(stopwords.words('english'))
for token in word_tokens:
    if token not in stop_words:
        result.append(token)
    """
    filter_setup_list = """
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
target = 'I graduated M.Eng. in CSE, but I want to pursue Ph.D. in CSE.'
word_tokens = word_tokenize(target)

result = list()
stop_words = stopwords.words('english')
for token in word_tokens:
    if token not in stop_words:
        result.append(token)
    """
    consumed_time = timeit.timeit(stmt=filter_setup_set, number=10000)
    print('consumed times for filtering with stop words as set:', consumed_time)
    consumed_time = timeit.timeit(stmt=filter_setup_list, number=10000)
    print('consumed times for filtering with stop words as list:', consumed_time)


def remove_stop_words_for_korean() -> None:
    okt = Okt()
    target_kor = '평일에 열심히 일하셨을거에요. 주말입니다. 이제 쉬죠.'
    stop_words = '에 하셨을거에요 입니다 죠'
    stop_words = set(stop_words.split(' '))
    work_tokens = okt.morphs(target_kor)
    result = [word for word in work_tokens if not word in stop_words]
    print('Before removal stop-words:', work_tokens)
    print('After removal stop-words:', result)


if __name__ == '__main__':
    # check_stop_words()
    # measure_filter_with_stop_words()
    remove_stop_words_for_korean()
