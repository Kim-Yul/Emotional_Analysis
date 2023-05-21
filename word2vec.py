from gensim.models import word2vec

def Word2vec(data):
    model = word2vec.Word2Vec(sentences = data, vector_size = 300, window = 5, min_count = 2, workers = 4)
    
    print(model.wv.most_similar("재미"))
    return model