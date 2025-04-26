# 건강검진 정보 데이터 전처리 파이프라인 (VSCode 실행용)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 데이터 불러오기 (EUC-KR 인코딩으로 한글 깨짐 방지)
df = pd.read_csv(r'E:/Midterm-Tea4/2번_public.health.CSV', encoding='euc-kr')

# 2. 데이터 구조 확인
print(df.shape)
print(df.columns)
print(df.head())

# 3. 결측치 시각화 및 처리
plt.figure(figsize=(12,6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()

# 결측치 비율 출력
missing_ratio = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
print("\n[결측치 비율 높은 순]")
print(missing_ratio[missing_ratio > 0])

# 간단한 결측치 처리 예시 (문자형은 'Unknown', 숫자형은 평균 또는 중간값으로 대체)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('Unknown')
    else:
        df[col] = df[col].fillna(df[col].median())

# 4. 이상치 탐지: 예시 - BMI, 수축기/이완기 혈압
numeric_cols = ['신장(5Cm단위)', '체중(5Kg단위)', '수축기혈압', '이완기혈압', 'BMI']
for col in numeric_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

# 5. 범주형 인코딩 (성별 등)
categorical_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 6. 정규화 (수치형 컬럼)
numeric_features = df.select_dtypes(include='number').drop(columns=['흡연상태'], errors='ignore')
scaler = MinMaxScaler()
df[numeric_features.columns] = scaler.fit_transform(df[numeric_features.columns])

# 7. 파생 변수 생성 예시: 고혈압 여부
if '수축기혈압' in df.columns and '이완기혈압' in df.columns:
    df['고혈압여부'] = ((df['수축기혈압'] >= 140) | (df['이완기혈압'] >= 90)).astype(int)
    print("파생 변수 '고혈압여부' 생성 완료")

# 8. 전처리 결과 저장
output_path = 'health_checkup_preprocessed.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"\n✅ 전처리된 데이터 저장 완료: {output_path}")

# 9. README용 요약 출력
total_rows = len(df)
missing_cols = df.isnull().sum().sum()
print("\n[README 요약문 예시 ⬇️]")
print(f"총 {total_rows}건의 건강검진 데이터를 전처리하였고, 결측치 및 이상치를 제거하였으며, 범주형 변수 인코딩과 정규화, 고혈압 여부 파생 변수를 생성하여 분석 가능한 형태로 정제했습니다.")
