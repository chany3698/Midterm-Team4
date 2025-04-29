import pandas as pd
import numpy as np
from sklearn.model_selection   import train_test_split
from sklearn.impute            import SimpleImputer
from sklearn.preprocessing     import StandardScaler, OneHotEncoder
from sklearn.compose           import ColumnTransformer
from sklearn.pipeline          import Pipeline

df = pd.read_csv('csvfile/7_heart.csv')
df.replace('?', np.nan, inplace=True)  # 데이터 로드 + 결측치 처리

for col in ['ca', 'thal']:
    df[col] = pd.to_numeric(df[col], errors='coerce') # 숫자 변환

X = df.drop(columns='condition') # 특성·타깃 분리
y = df['condition']

# 수치형 파이프라인: 중앙값 대체 → 표준화
numeric_features   = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

# 범주형 파이프라인: 최빈값 대체 → 원-핫
categorical_features   = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(drop='first', sparse_output=False))  # <-- 변경된 부분
])

# 통합 전처리기
preprocessor = ColumnTransformer([
    ('num', numeric_transformer,   numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

