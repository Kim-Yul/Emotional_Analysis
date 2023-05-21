import pickle
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pre_processing import *
from tokenizer import *
from word2vec import *
from int_encoding import *
from model import *

def below_threshold_len(max_len, nested_list):
    count = 0
    for sentence in nested_list:
        if(len(sentence) <= max_len):
            count = count + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (count / len(nested_list))*100))


# csv 파일 읽기 : train파일과 test 파일로 진행
train_data = pd.read_csv('ratings_train.txt', sep='\t')
test_data = pd.read_csv('ratings_test.txt', sep='\t')


# 훈련용 리뷰 데이터 개수 및 상위 5개 데이터 확인
print('훈련용 리뷰 개수 :', len(train_data))
#print(train_data[:5])

# 테스트용 리뷰 데이터 개수 및 상위 5개 데이터 확인
print("테스트용 리뷰 개수 :", len(test_data))
#print(test_data[:5])

"""
: 훈련용 리뷰 데이터 수는 150,000개,
  테스트용 리뷰 데이터 수는 50,000개
"""

# 데이터 전처리 과정
train_data = Pre_processing(train_data)
test_data = Pre_processing(test_data)


print(train_data.groupby('label').size().reset_index(name = 'count'))
print(test_data.groupby('label').size().reset_index(name = 'count'))

# KLT2000()
#X_train = KLT2023(train_data)
#X_test = KLT2023(test_data)

# OKT()
X_train = OKT(train_data)
X_test = OKT(test_data)

# 워드임베딩 = word2vec
X_train_model = Word2vec(X_train)
X_test_model = Word2vec(X_test)

# 정수 인코딩
X_train, X_test, y_train, y_test, vocab_size, tokenizer = Int_encoding(X_train, X_test, train_data, test_data)

print(X_train[:3])

""" # sentence 구조
for index, sentence in enumerate(X_train):
    if len(sentence) < 1:
        print(index)
        break
"""
drop_train = [index for index, sentence in enumerate(X_train) if len(sentence) < 1]

print("제거해야할 데이터의 수 : ", len(drop_train))
print("제거하기 전 데이터의 수 : ", len(X_train))

X_train = np.array(X_train, dtype=object)
X_train = np.delete(X_train, drop_train, axis=0)
y_train = np.array(y_train, dtype=object)
y_train = np.delete(y_train, drop_train, axis=0)

print(len(X_train))
print(len(y_train))

# 패딩
print('리뷰의 최대 길이 :',max(len(review) for review in X_train))
print('리뷰의 평균 길이 :',sum(map(len, X_train))/len(X_train))

max_len = 30
below_threshold_len(max_len, X_train)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

X_train = X_train.astype(float)
y_train = y_train.astype(float)


# LSTM으로 영화 리뷰 감상하기
embedding_dim = 100
hidden_units = 128

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim))
model.add(LSTM(hidden_units))
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=64, validation_split=0.4)

loaded_model = load_model('best_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

# 파일 저장
with open('tokenizer.pickle', 'wb') as handle:
     pickle.dump(tokenizer, handle)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


sentiment_predict('이 영화 개꿀잼 ㅋㅋㅋ')
sentiment_predict('이 영화 핵노잼 ㅠㅠ')
sentiment_predict('이딴게 영화냐 ㅉㅉ')
sentiment_predict('감독 뭐하는 놈이냐?')
sentiment_predict('와 개쩐다 정말 세계관 최강자들의 영화다')