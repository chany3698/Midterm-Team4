import pandas as pd
import numpy as np
from scipy import stats

# CSV 파일 불러오기
df = pd.read_csv("Dataset/6번_StudentsPerformance.csv")

#  문자열 공백 제거 + 결측치 'Unknown'으로 대체
df['parental level of education'] = df['parental level of education'].str.strip()
df['parental level of education'] = df['parental level of education'].fillna('Unknown')

#  재분류 맵 만들기
edu_map = {
    'some high school': 'High School',
    'high school': 'High School',
    'some college': 'College',
    "associate's degree": 'College',
    "bachelor's degree": 'University',
    "master's degree": 'University',
    "Unknown": 'Unknown'  # 결측치 대체용
}

#  그룹핑
df['parental_edu_group'] = df['parental level of education'].map(edu_map)

# 결측치 확인
print("결측치 확인:\n", df.isnull().sum())

# 이상치 제거 (Z-score 이용)
score_cols = ['math score', 'reading score', 'writing score']
z_scores = stats.zscore(df[score_cols])

# Z-score 절댓값이 3 미만인 행만 남김
df = df[(np.abs(z_scores) < 3).all(axis=1)]

# 범주형 변수 → pandas.get_dummies()로 인코딩
df_dummies = pd.get_dummies(df, columns=[
    'gender',
    'race/ethnicity',
    'parental level of education',
    'lunch',
    'test preparation course'
])

# 기준 열 수동 삭제
df_dummies.drop(columns=[
    'gender_female',
    'race/ethnicity_group A',
    "parental level of education_associate's degree",
    'lunch_free/reduced',
    'test preparation course_completed'
], inplace=True)

# 평균 점수 컬럼 추가
df_dummies['average_score'] = df_dummies[score_cols].mean(axis=1)

# 결과 저장
df_dummies.to_csv("Preprocessed_dataset/6번_StudentsPerformance_processed.csv", index=False)


