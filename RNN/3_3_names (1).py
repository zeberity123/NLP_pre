# 3_3_names.py
import nltk
import random
import string
import numpy as np


# nltk.download('names')

print(nltk.corpus.names.fileids())
print(nltk.corpus.names.words('female.txt'))
print(nltk.corpus.names.raw('female.txt'))


# 퀴즈
# 남자와 여자 이름을 아래 형식처럼 구성하세요
# 그리고
# 2개를 합쳐서 80%와 20%로 나누어 주세요
# [(Abagael, female), (Abbe, female), ...]
# [(Tom, male), (John, male), ...]
def make_labeled_names():
    females = [(n, 'female') for n in nltk.corpus.names.words('female.txt')]
    males = [(n, 'male') for n in nltk.corpus.names.words('male.txt')]
    # print(len(females), len(males))     # 5001 2943

    labeled = females + males
    random.shuffle(labeled)
    # print(labeled[:5])

    return labeled


def gender_classification(labeled, feature_func):
    # data = [({'last': name[-1]}, gender) for name, gender in labeled]
    data = [(feature_func(name), gender) for name, gender in labeled]
    # print(data[:5])

    train_set = data[:6000]
    test_set = data[6000:]

    clf = nltk.NaiveBayesClassifier.train(train_set)
    acc = nltk.classify.accuracy(clf, test_set)
    print(acc)

    # print(clf.classify({'last': 'Trinity'[-1]}))
    # print(clf.classify(make_feature_1('Trinity')))
    # print(clf.classify(feature_func('Trinity')))

    return acc


def make_feature_1(name):
    return {'last': name[-1]}


def make_feature_2(name):
    return {'last': name[-1], 'first': name[0]}


def make_feature_3(name):
    features = {'last': name[-1], 'first': name[0]}

    for ch in string.ascii_lowercase:       # 'abcdefghijklmnopqrstuvwxyz':
        features['count_{}'.format(ch)] = str.count(name, ch)
        features['has_{}'.format(ch)] = (ch in name)

    return features


# 퀴즈
# 여러분만의 피처 함수를 만드세요
def make_feature_4(name):
    return {'last1': name[-1], 'last2': name[-2]}


def make_feature_5(name):
    return {'last1': name[-1], 'last2': name[-2:]}


def make_feature_6(name):
    return {'last1': name[-1], 'last2': name[-2:], 'first': name[0], 'len': len(name)}


# 퀴즈
# 10번 반복해서 몇 번째가 제일 잘 하는지 알려주세요
bests = []
for i in range(10):
    labeled = make_labeled_names()

    scores = [
        gender_classification(labeled, make_feature_1),
        gender_classification(labeled, make_feature_2),
        gender_classification(labeled, make_feature_3),
        gender_classification(labeled, make_feature_4),
        gender_classification(labeled, make_feature_5),
        gender_classification(labeled, make_feature_6),
    ]
    
    print()

    bests.append(np.argmax(scores))

print('best :', bests)
