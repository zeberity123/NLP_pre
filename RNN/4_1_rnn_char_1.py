# 4_1_rnn_char_1.py
import keras

# tensor
# t -> e
# e -> n
# n -> s
# s -> o
# o -> r
# x: tenso
# y: ensor

# 아래 데이터에 대해 동작하는 모델

def rnn_char_1_onehot():

    x = [[1, 0, 0, 0, 0, 0],    # t
        [0, 1, 0, 0, 0, 0],    # e
        [0, 0, 1, 0, 0, 0],    # n
        [0, 0, 0, 1, 0, 0],    # s
        [0, 0, 0, 0, 1, 0]]    # o

    y = [[0, 1, 0, 0, 0, 0],    # e
        [0, 0, 1, 0, 0, 0],    # n
        [0, 0, 0, 1, 0, 0],    # s
        [0, 0, 0, 0, 1, 0],    # o
        [0, 0, 0, 0, 0, 1]]    # r

    model = keras.Sequential([
        keras.layers.Input(shape=(6,)),
        keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(0.1),
                loss=keras.losses.categorical_crossentropy,
                metrics='acc')
    
    model.fit(x, y, epochs=10, verbose=2)


# onehot을 sparse 버전으로 변환
def rnn_char_1_sparse():

    x = [[1, 0, 0, 0, 0, 0],    # t
        [0, 1, 0, 0, 0, 0],    # e
        [0, 0, 1, 0, 0, 0],    # n
        [0, 0, 0, 1, 0, 0],    # s
        [0, 0, 0, 0, 1, 0]]    # o

    y = [1, 2, 3, 4, 5] # e n s o r

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


# rnn_char_1_onehot()
rnn_char_1_sparse()
