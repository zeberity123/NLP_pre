import keras
import numpy as np

def rnn_char_2_sorted():
    # tensor -> enorst
    x = [[0, 0, 0, 0, 0, 1],  # t
         [1, 0, 0, 0, 0, 0],  # e
         [0, 1, 0, 0, 0, 0],  # n
         [0, 0, 0, 0, 1, 0],  # s
         [0, 0, 1, 0, 0, 0]]  # o
    y = [0, 1, 4, 2, 3]       # e n s o r

    model = keras.Sequential([
         keras.layers.Input(shape=(6,)),
         keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=10, verbose=2)

    p = model.predict(x, verbose=0)
    print(p)

def rnn_char_2_simple():

    vocab = sorted('tensor')
    vocab = np.array(vocab)     # ['e' 'n' 'o' 'r' 's' 't']

    # tensor -> enorst
    x = [[0, 0, 0, 0, 0, 1],  # t
         [1, 0, 0, 0, 0, 0],  # e
         [0, 1, 0, 0, 0, 0],  # n
         [0, 0, 0, 0, 1, 0],  # s
         [0, 0, 1, 0, 0, 0]]  # o
    y = [0, 1, 4, 2, 3]       # e n s o r

    x = [x]
    y = [y]

    model = keras.Sequential([
         keras.layers.Input(shape=(5, 6)),
         keras.layers.SimpleRNN(16, return_sequences=True),
         keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=10, verbose=2)

    # x, y를 3차원으로 변형했을 때의 에러를 수정
    p = model.predict(x, verbose=0)
    print(p)

    p_arg = np.argmax(p, axis=2) # [[0 1 4 2 3]]
    print('acc : ', np.mean(p_arg == y))

    # 예측결과 디코딩
    print([i for i in p_arg[0]])
    print([vocab[i] for i in p_arg[0]])
    print(vocab[p_arg[0]])
    print(vocab[p_arg])

# rnn_char_2_sorted()

rnn_char_2_simple()