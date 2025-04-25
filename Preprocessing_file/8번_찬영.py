import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


df = pd.read_csv("Dataset/8번_netflix_titles.csv")

# 컬럼 이름 변경
df = df.rename(columns={"listed_in": "Genres"})

# 장르 컬럼에서 "," 기준으로 분리하여 원 핫 인코딩 수행
genres_ohe = (
    df["Genres"]
    .str.get_dummies(sep=",")
)

# 각 열별로 앞뒤 공백 제거
genres_ohe.columns = genres_ohe.columns.str.strip()

# 겹치는 컬럼 합체, 값이 0과 1이였다면 1로
genre_ohe = genres_ohe.groupby(axis=1, level=0).sum()

# Genres 열 삭제해주고 genre ohe 형태로 열 생성
df_final = pd.concat([df.drop(columns=["Genres"]), genre_ohe], axis=1)

df_final.to_csv("Preprocessed_dataset/8번_netflix_titles_genres_ohe.csv", index=False)


# ------------------------------------------------------------------------------------
# 배우별 활동 시기 시각화
actor_name = input("name: ")        
mask = df["cast"].fillna("").str.contains(actor_name, case=False)


# 입력한 배우 이름이 cast 컬럼에 포함된 행만 필터링
df_actor = df[mask] 

# 연도별 편수 집계
year_counts = (
    df_actor
      .groupby("release_year")
      .size()                 # 편수 세기
      .sort_index()           # 연도 순으로 정렬
)

print(year_counts.head())   

# 막대 그래프로 시각화
plt.figure(figsize=(10, 4))
year_counts.plot(kind="bar")  
plt.title(f"Netflix Titles Featuring '{actor_name}' by Release Year")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.show()


# 선그래프로 시각화
plt.figure(figsize=(10, 4))
year_counts.plot(marker='o')   
plt.title(f"Netflix Titles Featuring '{actor_name}' by Release Year (Line)")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------------
# 시간대 별 시각화

#Movie 타입만 필터링
movies = df[df["type"] == "Movie"].copy()

# duration 문자열에서 숫자만 추출 → 숫자로 변환
movies["minutes"] = (
    pd.to_numeric(
        movies["duration"].str.extract(r"(\d+)").squeeze(),
        errors="coerce"          # 숫자 추출 실패 → NaN
    )
)
# 추출 실패(NaN) 행 제거, 정수로 변환
movies = movies.dropna(subset=["minutes"])
movies["minutes"] = movies["minutes"].astype(int)
# 구간(예: <60, 60–89 …)을 미리 정의
bins_min   = [0, 60, 90, 120, 180, 10_000]
labels_min = ["< 60", "60-89", "90-119", "120-179", "180+"]

movies["minutes_bin"] = pd.cut(movies["minutes"],
                               bins=bins_min,
                               labels=labels_min,
                               right=False)  # 오른쪽 경계 포함 X ⇒ 60은 두 번째 구간에 포함

# 구간별 작품 수 집계
min_counts = movies["minutes_bin"].value_counts().sort_index()

plt.figure(figsize=(7, 4))
min_counts.plot(kind="bar")
plt.title("Movie count by runtime (minutes)")
plt.xlabel("Runtime range (min)")
plt.ylabel("Count")
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.tight_layout()
plt.show()
