# 햄릿에 나오는 주인공 이름이 출현한 빈도를 막대그래프로
import nltk
import collections
import operator
import matplotlib.pyplot as plt

hamlet = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')
hamlet = hamlet.lower()
# print(hamlet)

tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(hamlet)
# print(tokens[:10])

# # print(nltk.corpus.stopwords.fileids())
# stop_words = nltk.corpus.stopwords.words('english')
# # print(stop_words)
# # print(len(tokens))

# # 퀴즈
# # 토큰 목록으로부터 불용어와 1글자 토큰을 제거하세요
# tokens = [t for t in tokens if t not in stop_words]
# print(len(tokens))

# tokens = [t for t in tokens if len(t) > 1]
# print(len(tokens))
# print(tokens)

names = ['hamlet', 'claudius', 'gertrude', 'polonius', 'ophelia', 'laertes', 'horatio']
tokens = [t for t in tokens if t in names]


freq = collections.defaultdict(int)
for t in tokens:
    freq[t] += 1

# print(freq)

# 퀴즈
# 빈도에 따라 정렬하세요 (리스트 생성)
# print(freq.items())
freq_sorted = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)

vocab = [t[0] for t in freq_sorted]
print(vocab)

freq = nltk.FreqDist(tokens)
print(freq)
print(freq.most_common(7))
freq_mc = freq.most_common(7)

x = [i[0] for i in freq_mc]
y = [i[1] for i in freq_mc]

print(x, y)

plt.bar(x, y)
plt.show()