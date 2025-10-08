import re

if __name__ == '__main__':
    text = 'I am studying NLP when I have free time'
    short_word = re.compile(r'\W*\b\w{1,2}\b')
    print(short_word.sub('', text))
    pass
