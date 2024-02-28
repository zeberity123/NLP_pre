# 4_7_stock.py
import pandas as pd
import numpy as np
import nltk
import keras
from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt


# 퀴즈
# stock_daily.csv 파일을 읽고
# RNN에 사용할 수 있는 x, y를 반환하는 함수를 만드세요
def make_xy(seq_length):
    # stock = np.loadtxt('data/stock_daily.csv', delimiter=',')
    # print(stock.shape)

    stock = pd.read_csv('data/stock_daily.csv',
                        skiprows=2, header=None)
    values = stock.values[::-1]

    # values = preprocessing.minmax_scale(values)

    scaler = preprocessing.MinMaxScaler()
    values = scaler.fit_transform(values)

    x, y = [], []
    for gram in nltk.ngrams(values, seq_length+1):
        x.append(gram[:-1])
        y.append(gram[-1][-1])

    return np.float32(x), np.float32(y), scaler


x, y, scaler = make_xy(seq_length=7)
print(x.shape, y.shape)     # (725, 7, 5) (725,)

print(scaler.data_max_)
print(scaler.data_min_)

data = model_selection.train_test_split(x, y, train_size=0.7, shuffle=False)
x_train, x_test, y_train, y_test = data

model = keras.Sequential([
    keras.layers.Input(shape=x.shape[1:]),
    keras.layers.SimpleRNN(16, return_sequences=True),
    keras.layers.SimpleRNN(16, return_sequences=False),
    keras.layers.Dense(1)
])
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.01),
              loss=keras.losses.mse,
              metrics='mae')

model.fit(x_train, y_train, epochs=10, verbose=2)

# 퀴즈
# stock_daily.csv 파일을 읽고
# 70%로 학습하고 30%에 대해 결과를 예측해서
# 그래프로 그려주세요
# v = (x - min) / (max - min)
# x = v * (max - min) + min

p = model.predict(x_test, verbose=0)

close_max = scaler.data_max_[-1]
close_min = scaler.data_min_[-1]

p = p * (close_max - close_min) + close_min
y_test = y_test * (close_max - close_min) + close_min

indices = np.arange(len(x_test))
plt.plot(indices, y_test, 'r', label='target')
plt.plot(indices, p, 'g', label='predict')
plt.legend()
plt.show()

# 퀴즈
# http://finance.yahoo.com/quote/GOOG/history?ltr=1
# 위 사이트로부터 구글 지주회사 알파벳 데이터를 다운로드 받아
# 여기서 만든 모델에 적용하세요

