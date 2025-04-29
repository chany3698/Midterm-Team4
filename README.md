# Midterm-Team4

# 1번 `1_adults.py`  
**– 미국 인구조사 소득 데이터 전처리 파이프라인**

미국 성인 소득 데이터(`1_adults.csv`)를 기반으로 결측치 처리, 인코딩, 정규화, 이상치 제거, 파생변수 생성 과정을 수행하여  
소득 `$50,000 이상`인 사람들만 필터링한 최종 전처리 데이터를 CSV로 저장합니다.

---

## 1. 입력 데이터

| 파일명 |
|--------|
| `모세/1_adults.csv` |

### 주요 컬럼 예시

| 컬럼명         | 설명                        |
|----------------|-----------------------------|
| `age`          | 나이                        |
| `sex`          | 성별 (남성/여성)            |
| `income`       | 소득 수준 (<=50K / >50K)    |
| `education.num`, `fnlwgt`, `capital.gain`, `hours.per.week` | 숫자형 변수들 |

---

## 2. 전처리 단계

| 단계 | 코드 라인 예시 | 설명 |
|------|----------------|------|
| **1) 패키지 임포트** | `import pandas as pd` 등 | pandas, numpy, matplotlib, seaborn, sklearn 등 불러오기 |
| **2) 파일 로드** | `pd.read_csv(...)` | CSV 파일 로드 |
| **3) 결측치 확인** | `df.isnull().sum()` | 결측치 개수 확인 |
| **4) 인코딩 (LabelEncoder)** | `le.fit_transform(df['sex'])` | 성별 `F/M`을 0/1로 변환 |
| **5) 문자열 → 숫자 변환** | `pd.to_numeric(..., errors='coerce')` | 숫자형 열 중 오류값을 NaN 처리 |
| **6) 결측치 종류 표준화** | `replace(..., pd.NA)` | '?', 'unknown' 등 → `pd.NA`로 통일 |
| **7) 결측치 채우기** | `fillna()` | 문자형: "Unknown" / 숫자형: 평균값 |
| **8) 정규화** | `MinMaxScaler` | `age` 컬럼을 0~1 사이 값으로 스케일링 |
| **9) 이상치 제거** | IQR 방식 | age 기준 상/하위 1.5IQR 벗어난 값 제거 |
| **10) 중복 제거** | `df.drop_duplicates()` | 완전 동일한 행 제거 |
| **11) 파생 변수 생성** | `income_level = ...` | 소득 수준을 새 변수로 정의 |
| **12) 고소득 필터링** | `df[df['income_level'] == '$50,000 이상']` | `$50,000 이상` 대상만 남김 |
| **13) 최종 결과 저장** | `to_csv(...)` | 전처리된 데이터 csv로 저장 |

---

## 3. 출력 파일

| 파일명 |
|--------|
| `adults_final_data.csv` |

---

해당 파이프라인은 결측치 처리부터 이상치 제거, 정규화, 인코딩, 파생 변수 생성 및 소득기준 필터링까지 포함하여  
모델 학습에 적합한 데이터로 가공하는 데 목적이 있습니다.


# 2번 `Credit Card Default Dataset`  
**— 신용카드 고객의 연체 가능성을 사전에 예측하여 위험 관리를 강화**

신용카드 고객의 정보(`2_Card.csv`)를 불러와 범주형 변수는 이진화와 매핑을 통해 해석 가능하도록 변환함
---

## 1. 입력 데이터

| 파일                | 설명                                                    |
|---------------------|---------------------------------------------------------|
| `2_Card.csv` | 신용카드 고객의 정보보 원본을 과제용 형식으로 제공한 CSV |

### 주요 컬럼

**수치형** : `LIMIT_BAL`, `AGE`, `PAY_0 ~ PAY_6`, `BILL_AMT1 ~ 6`, `PAY_AMT1 ~ 6`
**범주형** : `SEX`, `EDUCATION`, `MARRIAGE`  
**타깃**     : `default.payment.next.month` (1 = 연체 / 0 = 정상)

---

## 2. 전처리·파이프라인 단계

| 단계 | 설명 | 코드 라인 |
|------|------|-----------|
| **1) 컬럼명 가독성 향상** | 컬럼명 전체를 한글로 바꾸고 의미 명시 | `df.rename()` |
| **2) 성별 이진화** | `SEX`를 `남성`, `여성`으로 컬럼 생성성 | `(df['SEX'] == 1).astype(int)` |
| **3) 교육/결혼 상태 매핑** | 숫자로 되어 있는 코드를 의미있게 문자열로 변환환| `map({1: '대학원', …})` |
| **4) 원본 코드 컬럼 제거** | `SEX`,`EDUCATOIN`,`MARRIAGE` 삭제 | `df.drop(columns=[])`



# 3번 `뉴욕 숙박공유 데이터`  
**— 뉴욕시 지역별 예약 성공률과 수익성 예측, 호스트에게 가격 설정 가이드 제공**

지역별 숙소 데이터셋(`3_AB.csv`)을 불러와 호스트에게 가격 설정 과 운영 전략 가이드를 제공할 수 있도록 전처리를 시행!
---

## 1. 입력 데이터

| 파일                | 설명                                                    |
|---------------------|---------------------------------------------------------|
| `3_AB.csv` | 가격, 위치, 객실타입 등의 데이터셋 CSV |

### 주요 컬럼

**수치형** : `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `availability_365`
**범주형** : `room_type`, `neighbourhood_group`, `neighbourhood`  
**타깃**     : `availability_365`, `price × minimum_nights`

---

## 2. 전처리·파이프라인 단계

| 단계 | 설명 | 코드 라인 |
|------|------|-----------|
| **1) 컬럼명 가독성 향상** | 컬럼명 전체를 한글로 명시 | `df.rename()` |
| **2) 결측치 처리** | `last_review`에 있는 결측치는 `Unknown`으로 `reviews_per_month`는 `0` 전처리| `.fillna()`
| **3) 예약 가능성 파생변수 생성** | `availability_365 > 0`을 `1`, else `0`| `df['예약 가능성'] = ...`|
| **4) 수익성 파생변수 생성** | `price × minimum_nights`을 `예상 수익 (USD)` | `df['예상 수익'] = ...`

### 3. 결과 데이터의 구조
| 컬럼명 | 설명 | 코드 라인 |
| 예약 가능성 | 연간 예약 가능 일수 > 0 여부 (0 또는 1) |
| 예상 수익 (USD) | 가격 x 최소 숙박 일수 |
| 객실 유형, 지역 그룹 등 | 문자열 형태 그대로 유지 |



# 4번 `4_예빈.py`
**- 병원 예약 데이터(No-show dataset) 전처리 파이프라인**

환자 예약 데이터(`4_MED_NS.csv`)를 기반으로 결측치 처리, 중복 제거, 이상치 제거, 인코딩 및 날짜 특징 생성 과정을 수행하여 모델 학습용 csv 파일로 저장.

---

## 1. 입력 데이터

|파일|
|------|
|`예빈_Dataset/4_MED_NS.csv`|

### 주요 컬럼

|컬럼명|설명|
|--------|-----|
|`Gender`|성별(F,M)|
|ScheduleDay`.`AppointmentDay`|예약일, 진료일|
|`Scholarship`,`Hipertension`,`Diabetes`,`Alcoholism`,`Handcap`,`SMS_received`|건강 관련 특성(0/1)|
|`Neighbourhood`| 병원 위치 |
|`No-show`| 예약 불참 여부(Yes/No) |

---

## 2. 전처리 단계

| 단계 | 코드라인 | 설명 |
|------|----------|------|
| **1) 결측치 확인** | `missing_values = df.isnull().sum()` | 결측치 존재 여부 파악 |
| **2) 중복 제거** | `df_no_duplicates = df.drop_duplicates(subset=['PatientId'])` | `PatientId` 기준 중복 제거 |
| **3) 이상치 제거** | `Z-score` 활용 (`Age` 컬럼 기준) | `|Z| ≥ 3` 이상치 행 삭제 |
| **4) 이진 인코딩** | `Gender`, `No-show` 열 | F→0, M→1 / No→0, Yes→1로 변환 |
| **5) 범주 인코딩** | `Neighbourhood` 열 | `pandas.get_dummies()`로 원핫 인코딩 |
| **6) 날짜 특징 추가** | `ScheduledDay_weekday`, `AppointmentDay_weekday` 생성 | 요일(0=월~6=일) 추출 |


# 5번 `5_real_value.py`  
**— FIFA 선수의 *Expected Real Value* (ERV) 산출 파이프라인**

선수 데이터(`5_SOCCER.csv`)를 기반으로 선수의 계약기간, 신체·기술 스탯에
가중치를 부여하고 합산하여 
**실제 가치 기대치** 를 계산하고, 순위를 부여한 뒤 CSV로 저장.  
---
## 1. 입력 데이터

| 파일              | 
|-------------------|
| `csvfile/5_SOCCER.csv` |

### 주요 컬럼 (전처리 대상)

| 컬럼명              | 의미                                      |
|---------------------|-------------------------------------------|
| `contract_valid_until` | 계약 만료 연도                          |
| `age`               | 나이                                     |
| `physic`, `pace`, `shooting`, `power_stamina` | FIFA 능력치 (0 – 99) |
| `preferred_foot`    | ‘Left’ or ‘Right’                         |

---

## 2. 전처리·특징 엔지니어링 단계

| 단계 | 코드 라인 | 설명 |
|------|-----------|------|
| **1) 불필요한 열 제거** | `df.drop(...)` | URL, 풀네임 등 모델에 불필요한 4개 열 삭제 |
| **2) 남은 계약 기간**| `remaining_contract_years` | `contract_valid_until - base year(2021)` |
| **3) 나이 보정치**  | `age_factor` | `max(age) - age`<br>→ 나이가 어릴수록 보정 ↑ |
| **4) 기대몸값 계산**    | `real_value_base` | 아래 가중치를 곱해 합산 |
| **5) 왼발 보너스**  | `foot_bonus` | 왼발 선수 `1.1`⋅, 우/양발 `1.0` |
| **6) 최종 기대몸값**    | `real_value_expectation` | `real_value_base × foot_bonus` |
| **7) 순위 매김**    | `value_rank` | *dense rank* (동점 시 동일 순위) |

### 기본 가중치 (튜닝 가능)

```python
w_contract = 1.0   # 잔여 계약연수
w_age      = 1.0   # 나이 보정
w_physic   = 1.2   # 피지컬
w_pace     = 1.0   # 속도
w_shooting = 1.0   # 슈팅
w_stamina  = 0.8   # 체력
```


# 7번 `7_heart_preprocess.py`  
**— 심장병 예측용 데이터 전처리 & 파이프라인**

심장병 데이터셋(`7_heart.csv`)을 불러와 결측값 처리,  
수치형·범주형 특성 변환(스케일링 + 원-핫 인코딩)을 수행,  
`ColumnTransformer`-기반 **전처리 파이프라인**(`preprocessor`).  
이를 통해 `fit_transform()` 한 번으로 모델 학습에 바로 투입할 수 있는 **`X_processed`** 과
레이블 `y`를 얻어냄.

---

## 1. 입력 데이터

| 파일                | 설명                                                    |
|---------------------|---------------------------------------------------------|
| `csvfile/7_heart.csv` | UCI Heart-Disease 원본을 과제용 형식으로 제공한 CSV |

### 주요 컬럼

**수치형** : `age`, `trestbps`(혈압), `chol`(콜레스테롤), `thalach`(최고심박수), `oldpeak`, `ca`  
**범주형** : `sex`, `cp`(가슴통증 타입), `fbs`(공복혈당), `restecg`, `exang`, `slope`, `thal`  
**타깃**     : `condition` (1 = 심장병 / 0 = 정상)

---

## 2. 전처리·파이프라인 단계

| 단계 | 설명 | 코드 라인 |
|------|------|-----------|
| **1) 결측치 표준화** | `'?' → NaN`  치환 | `df.replace('?', np.nan, inplace=True)` |
| **2) 수치형 형 변환** | `'ca' · 'thal'` 를 숫자로 강제 변환 (`errors='coerce'`) | `pd.to_numeric()` |
| **3) 특성·타깃 분리** | `X = df.drop('condition')`, `y = df['condition']` |  |
| **4) 수치형 파이프라인** | `median` 대체 → `StandardScaler()` | `numeric_transformer` |
| **5) 범주형 파이프라인** | `most_frequent` 대체 → `OneHotEncoder(drop='first')` | `categorical_transformer` |
| **6) 통합 파이프라인** | `ColumnTransformer([('num', …), ('cat', …)])` | `preprocessor` |

> - 더미 변수 폭발 방지를 위해 `drop='first'` 옵션을 사용