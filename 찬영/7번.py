import pandas as pd
import numpy as np
from sklearn.model_selection   import train_test_split
from sklearn.impute            import SimpleImputer
from sklearn.preprocessing     import StandardScaler, OneHotEncoder
from sklearn.compose           import ColumnTransformer
from sklearn.pipeline          import Pipeline

df = pd.read_csv('csvfile/7_heart.csv')
df.replace('?', np.nan, inplace=True)

for col in ['ca', 'thal']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

X = df.drop(columns='condition')
y = df['condition']

numeric_features   = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_features   = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(drop='first', sparse_output=False))  # <-- 변경된 부분
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer,   numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

