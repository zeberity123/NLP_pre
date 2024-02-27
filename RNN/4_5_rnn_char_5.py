# 4_5_rnn_char_5.py
import keras
import numpy as np
from sklearn import preprocessing


def make_xy(words):
    # ['tensor', 'tea', 'desktop'] -> 'tensorteadesktop'
    bin = preprocessing.LabelBinarizer()
    bin.fit(list(''.join(words)))

    max_len = max([len(w) for w in words])

    x, y = [], []
    for w in words:
        # 'tea' -> 'tea----'
        w += '-' * (max_len - len(w))

        onehot = bin.transform(list(w))
        print(onehot, end='\n\n')
        x.append(onehot[:-1])
        y.append(np.argmax(onehot[1:], axis=1))

    return np.int32(x), np.int32(y), bin.classes_


def rnn_char_5(words):
    x, y, vocab = make_xy(words)
    print(x.shape, y.shape)     

    model = keras.Sequential([
        keras.layers.Input(shape=x.shape[1:]),
        keras.layers.SimpleRNN(16, return_sequences=True),
        keras.layers.Dense(x.shape[-1], activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=10, verbose=2)

    p = model.predict(x, verbose=0)
    p_arg = np.argmax(p, axis=2)
    print(p_arg)

    print('acc :', np.mean(p_arg == y))

    # 퀴즈
    # 필요없는 부분을 삭제하세요
    valids = [len(w) - 1 for w in words]
    print(*[vocab[pp[:vv]] for pp, vv in zip(p_arg, valids)], sep='\n')
    print([''.join(vocab[pp[:vv]]) for pp, vv in zip(p_arg, valids)])


# 퀴즈
# 길이가 다른 단어의 목록에 대해 동작하도록 수정하세요
rnn_char_5(['tensor', 'tea', 'desktop'])


