# 1_3_abalone.py
import keras
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection


# 퀴즈
# abalone.data 파일에 대해
# 70%로 학습하고 30%에 대해 결과를 구하세요
def read_abalone():
    abalone = pd.read_csv('data/abalone.data', header=None)
    print(abalone)

    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(abalone[0].values)
    print(y)

    x = abalone.values[:, 1:]
    x = np.float32(x)
    print(x.dtype)

    return x, y


def model_1(x_train, y_train, x_test):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.0001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=50, verbose=2)
    return model.predict(x_test, verbose=0)


def model_2(x_train, y_train, x_test, y_test, n0, n1, n2):
    model = keras.Sequential([
        keras.layers.Dense(n0, activation='relu'),
        keras.layers.Dense(n1, activation='relu'),
        keras.layers.Dense(n2, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])

    # model = keras.Sequential()
    # for n in layers:
    #     keras.layers.Dense(n, activation='relu'),
    # keras.layers.Dense(3, activation='softmax'),

    model.compile(optimizer=keras.optimizers.Adam(0.0001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=50, verbose=0)
    print(model.evaluate(x_test, y_test, verbose=0))
    return model.predict(x_test, verbose=0)


x, y = read_abalone()

x = preprocessing.scale(x)
# x = preprocessing.minmax_scale(x)

data = model_selection.train_test_split(x, y, train_size=0.7)
x_train, x_test, y_train, y_test = data

ensemble = np.zeros([len(y_test), 3])
for layers in [(128, 64, 16), (64, 32, 12), (64, 16, 8),
               (128,48, 16), (64, 16, 16), (128, 32, 32), (256, 64, 64)]:
    p = model_2(x_train, y_train, x_test, y_test, *layers)
    # print(p.shape)                        # (1254, 3)

    ensemble += p

p_arg = np.argmax(ensemble, axis=1)
print('acc :', np.mean(p_arg == y_test))
