from konlp.kma.klt2023 import klt2023
from tqdm import tqdm
from konlpy.tag import Okt

def KLT2023(data):
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    k = klt2023()
    X_data = []
    for sentence in tqdm(data['document']):
        sentence = str(sentence)
        #tokenized_sentence = k.morphs(sentence)
        tokenized_sentence = k.nouns(sentence)
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords]
        X_data.append(stopwords_removed_sentence)

    return X_data

def OKT(data):
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']
    okt = Okt()
    X_data = []
    for sentence in tqdm(data['document']):
        tokenized_sentence = okt.morphs(sentence, stem=True) # 토큰화
        stopwords_removed_sentence = [word for word in tokenized_sentence if not word in stopwords] # 불용어 제거
        X_data.append(stopwords_removed_sentence)
    return X_data

"""
k = klt2023()
simple_txt = "내 눈을 본다면 밤하늘의 별이 되는 기분을 느낄 수 있을 거야"

pos_tags = k.pos(simple_txt)
print(pos_tags)

pos_tags = k.morphs(simple_txt)
print(pos_tags)

pos_tags = k.nouns(simple_txt)
print(pos_tags)
"""