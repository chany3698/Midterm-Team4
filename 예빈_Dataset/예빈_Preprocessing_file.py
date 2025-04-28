import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("예빈_Dataset/4_예빈.py.csv")

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]


print(missing_values)

# 중복 제거 (PatientId 기준)
df_no_duplicates = df.drop_duplicates(subset=['PatientId'])

# 이상치 제거 (Z-score 이용)
score_cols = ['Age']
z_scores = stats.zscore(df[score_cols])
df = df[(np.abs(z_scores) < 3).all(axis=1)]


# Label Encoding 대신 수동 변환
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})

    
# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['Neighbourhood'])

#날짜 전처리
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['ScheduledDay_weekday'] = df['ScheduledDay'].dt.dayofweek
df['AppointmentDay_weekday'] = df['AppointmentDay'].dt.dayofweek

df_encoded.to_csv("예빈_Dataset/4번_예빈.py.csv", index=False)
