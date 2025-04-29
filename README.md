# Midterm-Team4

# 2번 `Credit Card Default Dataset`  
**— 신용카드 고객의 연체 가능성을 사전에 예측하여 위험 관리를 강화화**

심장병 데이터셋(`2_Card.csv`)을 불러와 범주형 변수는 이진화와 매핑을 통해 해석 가능하도록 변환함
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