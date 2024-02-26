import gensim

text = ['나는 너를 사랑해', '나는 너를 미워해']
token = [s.split() for s in text]
print(token)

# 나는: 0, (1,0), 0.7, (0.7, 0.3)
# 너는: 1, (0,1), 0.2, (0.2, 0.3)

embedding = gensim.models.Word2Vec(token, min_count=1, vector_size=5)
print(embedding)
print(embedding.wv.index_to_key)
print(embedding.wv.vectors)

embedding = gensim.models.Word2Vec(token, min_count=1, vector_size=5, sg=True)
print(embedding)
print(embedding.wv.index_to_key)
print(embedding.wv.vectors)