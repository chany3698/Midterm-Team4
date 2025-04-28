import pandas as pd        #판다스 삽입
import numpy as np                     #넘파이 삽입
import matplotlib.pyplot as plt              #그래프 기능 삽입
import seaborn as sns                                #그래프 시각화 기능 삽입
from sklearn.preprocessing import LabelEncoder, MinMaxScaler                   #인코딩 및 정규화 기능 삽입

df=pd.read_csv(r"E:/Midterm-Team4/모세/1_adults.csv")                       #csv 파일 로드
print(df.head())                  #상위 5개만 확인
print(df.isnull().sum())               #결측치 확인


le = LabelEncoder()              #인코딩 실행
df['sex'] = le.fit_transform(df['sex'])              #성별 인코딩을 통해 숫자로 구분

numeric_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']              #문자열을 숫자열로 바꿀 열

for col in numeric_cols:                                                      #반복문을 통해 문자열 숫자열로 변환
    df[col] = pd.to_numeric(df[col], errors='coerce')                        #숫자열로 변환


missing_like = ['ERROR', 'UNKNOWN', 'missing', 'Missing', 'N/A', 'na', 'NaN', "?"]                                      #결측치에 해당하는 값들 모음

df.replace(missing_like, pd.NA, inplace=True)                                       #결측치들을 정리하기 편하게 NA값으로 하나로 통일


for col in df.columns:                                                            #반복문 사용
    if df[col].dtype == 'object':                                      #if 사용
        df[col] = df[col].fillna('Unknown')                            #문자열 결측치 "Unknown"으로 채우기
    else:                                                              #else 사용
        df[col] = df[col].fillna(df[col].mean())                       #숫자열 결측치 평균값으로 채우기


scaler = MinMaxScaler()                                                                                   #정규화 기능 사용
df[['age']] = scaler.fit_transform(df[['age']])                     #'Quantity', 'Price Per Unit' 정규화 기능 발동

Q1 = df['age'].quantile(0.25)                                     #이상값 측정 (하위25%)
Q3 = df['age'].quantile(0.75)                                     #이상값 측정 (상위25%)
IQR = Q3 - Q1

# 이상치 조건 설정
condition = (df['age'] >= (Q1 - 1.5 * IQR)) & (df['age'] <= (Q3 + 1.5 * IQR))              #하위 및 상위 25%를 제외한 50%만 인정, 그 이외는 이상값으로 처리
df = df[condition]

df.drop_duplicates(inplace=True)

df['income_level'] = df['income'].apply(lambda x: '$50,000 미만' if x == '<=50K' else '$50,000 이상')                                         #파생변수 생성

condition2 = (df['income_level'] >= (Q1 - 1.5 * IQR)) & (df['income_level'] <= (Q3 + 1.5 * IQR))              #하위 및 상위 25%를 제외한 50%만 인정, 그 이외는 이상값으로 처리
df = df[condition2]

print(df.isnull().sum())                                                       #결측치 확인
print(df.head())                                  #상단 5개 정보만 확인



output_path = 'adults_final_data.csv'                              #csv 파일로 업로드하기 위해 저장
df.to_csv(output_path, index=False, encoding='utf-8-sig')                  #csv 파일로 업로드