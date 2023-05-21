import numpy as np

def Pre_processing(data):
    data = data.copy()

    # document 열에 중복된 내용 제거
    data['document'].nunique(), data['label'].nunique()
    data.drop_duplicates(subset=['document'], inplace=True)

    if(data.isnull().values.any()) == True:
        data = data.dropna(how='any')

    # 한글 및 공백만 남기고 모두 지우기
    data['data'] = data['document'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
    
    # 공백을 empty 값으로 변경
    data['data'] = data['data'].str.replace('^ +', "")

    # 공백은 null 값으로 변경
    data['data'].replace('', np.nan, inplace=True) 
    
    data = data.dropna(how='any')
    
    # NULL 값 샘플 제거
    if(data.isnull().values.any()) == True:
        data = data.dropna(how='any')

    print('전처리 후 남은 샘플의 개수 : ', len(data))

    return data

