import pickle
import re
from konlpy.tag import Okt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sentiment_predict(new_sentence):
    # 설정
    okt = Okt()
    max_len = 30
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    loaded_model = load_model('best_model.h5')
    stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']


    # 문장 리뷰 확인
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
    new_sentence = okt.morphs(new_sentence, stem=True) # 토큰화
    new_sentence = [word for word in new_sentence if not word in stopwords] # 불용어 제거
    encoded = tokenizer.texts_to_sequences([new_sentence]) # 정수 인코딩
    #print(encoded)
    pad_new = pad_sequences(encoded, maxlen = max_len) # 패딩
    #print(pad_new)
    score = float(loaded_model.predict(pad_new)) # 예측
    #print(score)
    if(score > 0.5):
        print(new_sentence)
        print("{:.2f}% 확률로 긍정 리뷰입니다.\n".format(score * 100))
    else:
        print(new_sentence)
        print("{:.2f}% 확률로 부정 리뷰입니다.\n".format((1 - score) * 100))