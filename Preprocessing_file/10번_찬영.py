import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 데이터 읽기
df = pd.read_csv("Dataset/10번_survey.csv")

# Gender 정규화 → male / female / other
def normalize_gender(x: str) -> str:
    """
    주어진 성별 입력을 male / female / other 로 표준화
    """
    if not isinstance(x, str):
        return "other"

    t = x.strip().lower()  # 앞뒤 공백 제거, 소문자
    # male 패턴
    if any(k in t for k in ["male", "m", "cis male", "man"]):
        return "male"
    # female 패턴
    if any(k in t for k in ["female", "f", "cis female", "woman", "fem"]):
        return "female"
    return "other"

df["gender_norm"] = df["Gender"].apply(normalize_gender)
df["gender_sorted"] = df["gender_norm"]
# (선택) 분포 확인
print(df["gender_norm"].value_counts())

# Gender 원 핫 인코딩
gender_ohe = pd.get_dummies(df["gender_sorted"], prefix="gender")
df = pd.concat([df, gender_ohe], axis=1)

# 4) No. of employees 구간(Bin) 만들기
# 예: "1-5", "6-25" … 처럼 이미 문자열 범위라면 그대로 사용
# 숫자만 있을 경우 구간을 직접 자른다
if df["no_employees"].dtype == "object":
    emp_bin = df["no_employees"].str.strip()
else:
    # 숫자형 → 직접 구간 지정 (예: <10, 10-25, 26-100, 100+)
    bins_emp   = [0, 10, 25, 100, 10_000]
    labels_emp = ["<10", "10-25", "26-100", "100+"]
    emp_bin = pd.cut(df["no_employees"], bins=bins_emp, labels=labels_emp, right=False)

emp_counts = emp_bin.value_counts().sort_index()

# Leave 항목 분포
leave_counts = df["leave"].value_counts()

# 6) 시각화
#     규칙: matplotlib 단독, 한 그래프씩, 색상 지정 X

## (a) 직원 규모 막대그래프
plt.figure(figsize=(7, 4))
ax = emp_counts.plot(kind="bar", rot=0)
plt.title("Count by Company Size")
plt.xlabel("Employee range")
plt.ylabel("Count")
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
for p in ax.patches:                         # 막대 위 숫자
    ax.annotate(f"{int(p.get_height())}",
                (p.get_x() + p.get_width()/2, p.get_height()),
                ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.show()

## (b) 휴가(leave) 항목 막대그래프
plt.figure(figsize=(8, 4))
leave_counts.plot(kind="bar")
plt.title("Leave Policy Responses")
plt.xlabel("Leave response")
plt.ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 전처리 결과 저장 
df.to_csv("Preprocessed_dataset/10번_survey_clean.csv", index=False)