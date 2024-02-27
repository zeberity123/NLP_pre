# 4_3_rnn_char_3.py
import keras
import numpy as np
from sklearn import preprocessing


# 퀴즈
# 앞의 코드를 모든 단어에 대해 동작하도록 수정하세요
def make_xy(word):
    bin = preprocessing.LabelBinarizer()
    onehot = bin.fit_transform(list(word))
    print(onehot)

    x = onehot[:-1]
    y = np.argmax(onehot[1:], axis=1)

#     vocab = sorted(word)
#     vocab = np.array(vocab)     # ['e' 'n' 'o' 'r' 's' 't']

#     return x[np.newaxis], y[np.newaxis], vocab
    # x[np.newaxis, :, :]
    # x[:, np.newaxis, :]
    # x[:, :, np.newaxis]
    # return x[np.newaxis], y[np.newaxis], bin.classes_
    return x[np.newaxis], y[np.newaxis], bin.classes_


def rnn_char_3(word):
    x, y, vocab = make_xy(word)
    print(x.shape, y.shape)     # (1, 5, 6) (1, 5)

    model = keras.Sequential([
        keras.layers.Input(shape=x.shape[1:]),
        keras.layers.SimpleRNN(16, return_sequences=True),
        keras.layers.Dense(x.shape[-1], activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=10, verbose=2)

    # 퀴즈
    # x, y를 3차원으로 변경했을 때의 에러를 수정하세요
    p = model.predict(x, verbose=0)
#     print(p)

    p_arg = np.argmax(p, axis=2)
    print(p_arg)                # [[0 1 4 2 3]]

    print('acc :', np.mean(p_arg == y))

    # 퀴즈
    # 예측 결과를 디코딩하세요
    print([i for i in p_arg[0]])
    print([vocab[i] for i in p_arg[0]])
    print(vocab[p_arg[0]])
    print(vocab[p_arg])


# rnn_char_3('tensor')
# rnn_char_3('desktop')
rnn_char_3('deep learning')



