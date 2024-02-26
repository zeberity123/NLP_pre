# 3_4_reviews.py
import nltk
import collections


# nltk.download('movie_reviews')

# print(nltk.corpus.movie_reviews.fileids())
# print(nltk.corpus.movie_reviews.categories())
# print(nltk.corpus.movie_reviews.fileids('neg'))
# print(nltk.corpus.movie_reviews.raw())

# 퀴즈
# 영화 리뷰 데이터로부터
# 80%로 학습하고 20%에 대해 정확도를 구하세요
fileids = nltk.corpus.movie_reviews.fileids()
total = [nltk.corpus.movie_reviews.raw(name) for name in fileids]
# print(total[1])
print(len(total))

raw = ''.join(total)
# print(len(raw))
print(len(nltk.corpus.movie_reviews.raw()))

tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(raw)
print(len(tokens))

stop_words = nltk.corpus.stopwords.raw('english')

tokens = [w for w in tokens if w not in stop_words]
tokens = [w for w in tokens if len(w) > 1]
print(len(tokens))

# freq = nltk.FreqDist(tokens)
freq = collections.Counter(tokens)
most2000 = freq.most_common(2000)
print(most2000[:5])

