import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("예빈_Dataset/4_예빈.py.csv")

# 결측치 확인
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]


print(missing_values)

# 중복 제거 (PatientId 기준)
df_no_duplicates = df.drop_duplicates(subset=['PatientId'])

# 이상치 제거 (Age컬럼에서 Z-score ±3을 벗어나는 값 제거)
score_cols = ['Age']
z_scores = stats.zscore(df[score_cols])
df = df[(np.abs(z_scores) < 3).all(axis=1)]


# 이진 인코딩
# 성별: F → 0, M → 1
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
# 예약 불참 여부: No → 0, Yes → 1
df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})

    
# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['Neighbourhood'])

#날짜 전처리(요일 정보 파생 변수 생성)
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay']) # 예약일 문자열 → 날짜형
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay']) # 진료일 문자열 → 날짜형
df['ScheduledDay_weekday'] = df['ScheduledDay'].dt.dayofweek # 예약일의 요일 (0=월 ~ 6=일)
df['AppointmentDay_weekday'] = df['AppointmentDay'].dt.dayofweek # 진료일의 요일

#결과 저장
df_encoded.to_csv("예빈_Dataset/4번_예빈.py.csv", index=False)
