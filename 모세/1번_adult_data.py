import pandas as pd        #판다스 삽입
import numpy as np                     #넘파이 삽입
import matplotlib.pyplot as plt              #그래프 기능 삽입
import seaborn as sns                                #그래프 시각화 기능 삽입
from sklearn.preprocessing import LabelEncoder, MinMaxScaler                   #인코딩 및 정규화 기능 삽입

df=pd.read_csv("1번_adult_data.csv", encoding="euc-kr")                       #csv 파일 로드
print(df.head())                  #상위 5개만 확인
print(df.isnull().sum())               #결측치 확인


le = LabelEncoder()              #인코딩 실행
df['sex'] = le.fit_transform(df['sex'])              #성별 인코딩을 통해 숫자로 구분

numeric_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']              #문자열을 숫자열로 바꿀 열

for col in numeric_cols:                                                      #반복문을 통해 문자열 숫자열로 변환
    df[col] = pd.to_numeric(df[col], errors='coerce')                        #숫자열로 변환
