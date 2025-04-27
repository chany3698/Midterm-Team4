import pandas as pd
import numpy as np
from scipy import stats

# 데이터 불러오기
df = pd.read_csv("Dataset/7번_dirty_cafe_sales.csv")

# 1. Item 결측치 -> 'Unspecified'
df['Item'] = df['Item'].fillna('Unspecified')

# 2. Payment Method, Location 결측치 -> 'Unknown'
df['Payment Method'] = df['Payment Method'].fillna('Unknown')
df['Location'] = df['Location'].fillna('Unknown')

# 3. 숫자형 필드 결측치 채우기
df['Quantity'] = df['Quantity'].fillna(0)
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['Price Per Unit'] = pd.to_numeric(df['Price Per Unit'], errors='coerce')
df['Price Per Unit'] = df['Price Per Unit'].fillna(df['Price Per Unit'].median())

# Total Spent 결측치는 계산해서 채우기
df['Total Spent'] = df['Total Spent'].fillna(df['Quantity'] * df['Price Per Unit'])

# 4. Transaction Date 결측치 삭제
df = df.dropna(subset=['Transaction Date'])

# 5. 중복 제거 (Transaction ID 기준)
df_no_duplicates = df.drop_duplicates(subset=['Transaction ID'])

# 6. 'Total Spent' 숫자 변환 (문자 오류 제거)
df_no_duplicates['Total Spent'] = pd.to_numeric(df_no_duplicates['Total Spent'], errors='coerce')

# 7. 변환 후 NaN 제거
df_no_duplicates = df_no_duplicates.dropna(subset=['Total Spent'])

# 8. 이상치 제거 (Z-score ±3 기준)
z_scores = stats.zscore(df_no_duplicates[['Total Spent']])
df_no_outliers = df_no_duplicates[(np.abs(z_scores) < 3).all(axis=1)]

# 9. 문자형 컬럼 인코딩 (Payment Method, Location, Item)
df_encoded = pd.get_dummies(df_no_outliers, columns=['Payment Method', 'Location', 'Item'])

# 최종 저장
df_encoded.to_csv("Preprocessed_dataset/7번_dirty_cafe_sales_processed.csv", index=False)

print("\n7번 전처리 완료: 결측치 처리 + 중복 제거 + 이상치 제거 + 인코딩 완료 후 저장했습니다.")
