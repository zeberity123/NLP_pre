# 5_3_word2vec_basic.py

# 주변단어를 추출하는 함수를 만드세요
def extract_surrounds(token_count, center, window_size):
    # for i in range(window_size):
    #     print(tokens[center-window_size+i])
    # for i in range(window_size):
    #     print(tokens[center+window_size+i-1])

    start = max(center - window_size, 0)
    end = min(center + window_size + 1, token_count)
    return [i for i in range(start, end) if i != center]

def show_dataset(tokens, window_size, is_skipgram):
    token_count = len(tokens)
    for center in range(token_count):
        surrounds = extract_surrounds(token_count, center, window_size)
        # print(center, surrounds)

        if is_skipgram:
            print(*[(tokens[center], tokens[i]) for i in surrounds])
        else:
            print([tokens[i] for i in surrounds], tokens[center])

#   0      1        2        3      4        5      6       7       8
# ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
sentence = "the quick brown fox jumps over the lazy dog"
tokens = sentence.split()
print(*tokens)

show_dataset(tokens, 2, 1)

# print(extract_surrounds(len(tokens), center=3, window_size=2))
# print(extract_surrounds(len(tokens), center=0, window_size=2))
# print(extract_surrounds(len(tokens), center=8, window_size=2))
