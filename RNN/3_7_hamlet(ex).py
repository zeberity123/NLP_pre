# 3_7_hamlet.py
import nltk
import collections
import matplotlib.pyplot as plt
from matplotlib import colors

# 퀴즈
# 세익스피어의 햄릿에 나오는 주인공 이름이 출현한 빈도를
# 막대 그래프로 그려주세요
print(nltk.corpus.gutenberg.fileids())

hamlet = nltk.corpus.gutenberg.raw('shakespeare-hamlet.txt')
hamlet = hamlet.lower()
tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(hamlet)

# freq = collections.Counter(tokens)
# print(freq.most_common(10))
# print(freq['hamlet'])

freq = collections.defaultdict(int)
for t in tokens:
    freq[t] += 1

print(freq['hamlet'])

actors = ['hamlet', 'ophelia', 'gertrude', 'claudius', 'laertes']
counts = [freq[n] for n in actors]
print(counts)

plt.bar(actors, counts, color=colors.TABLEAU_COLORS)
plt.show()



