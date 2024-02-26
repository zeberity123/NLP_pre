# 3_4_reviews.py
import nltk


# nltk.download('movie_reviews')

print(nltk.corpus.movie_reviews.fileids())
print(nltk.corpus.movie_reviews.categories())
print(nltk.corpus.movie_reviews.fileids('neg'))
print(nltk.corpus.movie_reviews.raw())

# 퀴즈
# 영화 리뷰 데이터로부터
# 80%로 학습하고 20%에 대해 정확도를 구하세요


