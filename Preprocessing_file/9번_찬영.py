import pandas as pd
import plotly.express as px

# 데이터 불러오기
df = pd.read_csv("Dataset/9번_cwurData.csv")

# 커스텀 가중치 설정
weights = {
    "quality_of_education": 0.40,
    "national_rank":        0.15,
    "alumni_employment":    0.05,
    "quality_of_faculty":   0.05,
    "publications":         0.15,
    "influence":            0.10,
    "citations":            0.10,
}

# 각 지표를 0~1 범위로 정규화 → 낮은 순위(작은 숫자)가 높게(1) 되도록 반전
#  norm = (max + 1 - value) / max
for col in weights.keys():
    max_val = df[col].max()
    df[f"norm_{col}"] = (max_val + 1 - df[col]) / max_val

# 가중 합계로 'my_score' 계산
df["my_score"] = sum(
    df[f"norm_{col}"] * w for col, w in weights.items()
)

# 내림차순 정렬 → 'my_rank' 부여
df = df.sort_values("my_score", ascending=False)
df["my_rank"] = range(1, len(df) + 1)

# 결과 확인 및 저장
print(df[["institution", "country", "my_rank", "my_score"]].head(10))

df.to_csv("Preprocessed_dataset/9번_cwurData_myrank.csv", index=False)

# 국가별 학교 수 집계 → 세계지도 히트맵
country_counts = (
    df["country"]
      .value_counts()
      .rename_axis("country")
      .reset_index(name="count")
)

fig = px.choropleth(
    country_counts,
    locations="country",
    locationmode="country names",   # 국가명 직접 매핑
    color="count",
    color_continuous_scale="Reds",
    title="Number of Universities per Country in CWUR Dataset",
)
fig.show()