import pandas as pd
import numpy as np
import nltk
import keras
from sklearn import preprocessing, model_selection

# stock_daily.csv 파일을 읽고
# RNN에 사용할 수 있는 x, y를 반환하는 함수

def make_xy(seq_length):
    stock = pd.read_csv('data/stock_daily.csv',
                        skiprows=2, header=None)
    
    values = stock.values

    
    
    x, y = [], []
    for gram in nltk.ngrams(values, seq_length+1):
        x.append(gram[:-1])
        y.append(gram[-1][-1])

    return np.float32(x), np.float32(y)

x, y = make_xy(seq_length=7)
print(x.shape, y.shape)

model = keras.Sequential([
    keras.layers.Input(shape=x.shape[1:]),
    keras.layers.SimpleRNN(16, return_sequences=False),
    keras.layers.Dense(1)
])

model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.001),
                loss=keras.losses.mse,
                metrics='mae')

model.fit(x, y, epochs=10, verbose=2)