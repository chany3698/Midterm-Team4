# 서울시 지하철역별 승하차 인원 정보 전처리 (압축 저장 포함)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 1. CSV 파일 로딩 (EUC-KR 인코딩 적용)
file_path = "서울시 지하철호선별 역별 승하차 인원 정보.csv"
df = pd.read_csv(file_path, encoding='euc-kr')

# 2. 컬럼명 정리
df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(명)", "", regex=False)
df.rename(columns={
    '지하철역': '역명',
    '호선명': '호선명',
    '승차총승객수': '승차수',
    '하차총승객수': '하차수',
    '사용일자': '사용일자'
}, inplace=True)

# 3. 날짜 컬럼 datetime 변환
df['사용일자'] = pd.to_datetime(df['사용일자'], format='%Y%m%d')

# 4. 결측치 처리
df.fillna(0, inplace=True)

# 5. 이상치 제거 (IQR 방식)
for col in ['승차수', '하차수']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# 6. 파생 변수: 총이용객수, 혼잡도
df['총이용객수'] = df['승차수'] + df['하차수']
scaler = MinMaxScaler()
df['혼잡도'] = scaler.fit_transform(df[['총이용객수']])

# 7. 범주형 인코딩
df['호선코드'] = LabelEncoder().fit_transform(df['호선명'])
df['역코드'] = LabelEncoder().fit_transform(df['역명'])

# 8. 분석용 최소 컬럼만 유지 (용량 축소 목적)
df_final = df[['사용일자', '호선코드', '역코드', '승차수', '하차수', '총이용객수', '혼잡도']]

# 9. gzip 압축 저장
output_path = "seoul_subway_cleaned.csv.gz"
df_final.to_csv(output_path, index=False, encoding='utf-8-sig', compression='gzip')
print(f"✅ 압축된 CSV 파일 저장 완료: {output_path}")

# 10. README용 요약 출력
print(f"총 {len(df_final)}건 전처리 완료, 혼잡도 파생변수 포함, gzip 압축 파일로 저장됨.")
