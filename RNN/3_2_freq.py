# 3_2_freq.py
import nltk
import collections
import operator

# 퀴즈
# webtext 코퍼스에 있는 wine.txt 파일을 토큰으로 만드세요
# nltk.download('stopwords')
print(nltk.corpus.webtext.fileids())

wine = nltk.corpus.webtext.raw('wine.txt')
wine = wine.lower()
print(wine)

tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(wine)
print(tokens[:10])

print(nltk.corpus.stopwords.fileids())
stop_words = nltk.corpus.stopwords.words('english')
print(stop_words)
print(len(tokens))

# 퀴즈
# 토큰 목록으로부터 불용어와 1글자 토큰을 제거하세요
tokens = [t for t in tokens if t not in stop_words]
print(len(tokens))

tokens = [t for t in tokens if len(t) > 1]
print(len(tokens))
print(tokens)

# 퀴즈
# 단어별 빈도를 딕셔너리에 저장하세요
# freq = {}
# for t in tokens:
#     if t in freq:
#         freq[t] += 1
#     else:
#         freq[t] = 1

# freq = {}
# for t in tokens:
#     if t not in freq:
#         freq[t] = 0
#
#     freq[t] += 1

# freq = {}
# for t in tokens:
#     freq[t] = freq.get(t, 0) + 1

freq = collections.defaultdict(int)
for t in tokens:
    freq[t] += 1

print(freq)

# 퀴즈
# 빈도에 따라 정렬하세요 (리스트 생성)
print(freq.items())
# freq_sorted = sorted(freq.items(), key=lambda t: t)
# freq_sorted = sorted(freq.items(), key=lambda t: t[1], reverse=True)
freq_sorted = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)
print(freq_sorted[:5])

vocab = [t[0] for t in freq_sorted[:2000]]
print(vocab[:10])

freq = nltk.FreqDist(tokens)
print(freq)
print(freq.most_common(10))

