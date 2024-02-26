# 3_1_nltk.py
import nltk         # natural language tool-kit
import re


def datasets():
    nltk.download('gutenberg')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('reuters')
    nltk.download('punkt')

    # nltk.download()


def corpus():
    print(nltk.corpus.gutenberg)
    print(nltk.corpus.gutenberg.fileids())

    emma = nltk.corpus.gutenberg.raw('austen-emma.txt')
    print(emma[:100])

    words = nltk.corpus.gutenberg.words('austen-emma.txt')
    print(words)

    # 퀴즈
    # emma 텍스트로부터 단어만 추출하세요
    print(re.findall(r'\w+', emma)[:10])
    print(re.findall(r'[a-zA-Z]+', emma)[:10])


def tokenize():
    emma = nltk.corpus.gutenberg.raw('austen-emma.txt')
    emma = emma[:1000]

    print(nltk.tokenize.simple.SpaceTokenizer().tokenize(emma))
    print(nltk.tokenize.sent_tokenize(emma))

    print(nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(emma))
    print(nltk.tokenize.WordPunctTokenizer().tokenize(emma))


# 어간 추출
# 우리는, 우리의, 우리가, 우리처럼, 우리랑, 우리만, ...
def stemming():
    words = ['lives', 'die', 'flies', 'died']

    st = nltk.stem.PorterStemmer()
    print(st.stem('lives'))

    # 퀴즈
    # words 리스트에 대해 어간을 추출하세요
    print([st.stem(w) for w in words])

    st = nltk.stem.LancasterStemmer()
    print([st.stem(w) for w in words])


def grams():
    text = 'hoon works at carrot' + ' ' + 'i like carrot'

    # 퀴즈
    # text를 단어 2개씩으로 묶어주세요
    # [(hoon, works), (works, at), (at, carrot)]
    # tokens = text.split()
    tokens = nltk.tokenize.word_tokenize(text)
    print([(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)])
    print([tokens[i:i+2] for i in range(len(tokens)-1)])

    print(list(nltk.bigrams(tokens)))
    print(list(nltk.trigrams(tokens)))
    print(list(nltk.ngrams(tokens, 4)))

nltk.download('punkt')
# datasets()
# corpus()
# tokenize()
# stemming()
grams()
