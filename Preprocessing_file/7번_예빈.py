import pandas as pd
import numpy as np
from scipy import stats

# 1. CSV 파일 불러오기
df = pd.read_csv("Dataset/7번_dirty_cafe_sales.csv")

# 2. 결측치 개수 확인
print("=== 원본 데이터 ===")
print(df.head())
print("\n결측치 개수 확인:")
print(df.isnull().sum())

# 3. 핵심 컬럼 기준 결측치 제거
core_cols = ['Transaction ID', 'Item', 'Total Spent']
df_no_na = df.dropna(subset=[col for col in core_cols if col in df.columns])
print("\n=== 결측치 삭제 후 ===")
print(df_no_na.head())

# 4. 나머지 결측치 대체
df_fill = df_no_na.copy()
if 'Payment Method' in df_fill.columns:
    df_fill['Payment Method'].fillna('Unknown', inplace=True)
if 'Location' in df_fill.columns:
    df_fill['Location'].fillna('Unspecified', inplace=True)

# 5. 중복 제거 (Transaction ID 기준)
df_no_duplicates = df_fill.drop_duplicates(subset=['Transaction ID'])

# 6. 'Total Spent'을 숫자로 변환 (문자형 오류 제거: 예. 'ERROR' → NaN)
df_no_duplicates['Total Spent'] = pd.to_numeric(df_no_duplicates['Total Spent'], errors='coerce')

# 7. 변환 후 NaN 제거
df_no_duplicates = df_no_duplicates.dropna(subset=['Total Spent'])

# 8. 이상치 제거 (Z-score 기준 ±3)
z_scores = stats.zscore(df_no_duplicates[['Total Spent']])
df_no_outliers = df_no_duplicates[(np.abs(z_scores) < 3).all(axis=1)]

# 9. 전처리 결과 저장
df_no_outliers.to_csv("Preprocessed_dataset/7번_dirty_cafe_sales_processed.csv", index=False)

