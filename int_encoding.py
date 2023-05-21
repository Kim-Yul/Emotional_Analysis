from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

def Int_encoding(data, tData, train_data, test_data):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data)

    #print(len(tokenizer.word_index))

    # 빈도 수
    threshold = 3
    # 단어의 수
    total_cnt = len(tokenizer.word_index)
    # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    rare_cnt = 0
    # 훈련 데이터의 전체 단어 빈도수 총 합 
    total_freq = 0
    # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합
    rare_freq = 0

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if(value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value


    print('단어 집합(vocabulary)의 크기 :',total_cnt)
    print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
    print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
    print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

    vocab_size = total_cnt - rare_cnt + 1
    print('단어 집합의 크기 :',vocab_size)

    tokenizer = Tokenizer(vocab_size) 
    tokenizer.fit_on_texts(data)
    X_train = tokenizer.texts_to_sequences(data)
    X_test = tokenizer.texts_to_sequences(tData)

    y_train = np.array(train_data['label'], dtype=np.int32)
    y_test = np.array(test_data['label'], dtype=np.int32)

    return X_train, X_test, y_train, y_test, vocab_size, tokenizer