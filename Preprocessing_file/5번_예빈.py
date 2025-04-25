import pandas as pd

# CSV 파일 불러오기
df = pd.read_csv("Dataset/5번_titanic/train.csv")

# Age 결측치 → 중간값으로 대체
df['Age'].fillna(df['Age'].median(), inplace=True)

# Cabin 컬럼 삭제
df.drop(columns=['Cabin'], inplace=True)

# Embarked 결측치 → 최빈값으로 대체 (새로운 컬럼 생성)
most_freq = df['Embarked'].mode()[0]
df['Embarked_fill'] = df['Embarked'].fillna(most_freq)

# 성별 인코딩 (LabelEncoder 없이 딕셔너리 매핑)
df['Sex_label'] = df['Sex'].map({'male': 0, 'female': 1})

# One-Hot Encoding: Embarked_fill 기준으로 인코딩
df_onehot = pd.get_dummies(df, columns=['Embarked_fill'], prefix='Embarked')

# 불필요한 열 제거
df_onehot.drop(columns=['Name', 'Ticket', 'PassengerId', 'Sex', 'Embarked'], inplace=True)

# 결과 저장
df_onehot.to_csv("Preprocessed_dataset/titanic_cleaned.csv", index=False)