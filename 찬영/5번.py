import pandas as pd

# 1) 원본 불러오기 & 불필요 컬럼 제거
df = pd.read_csv('csvfile/5_SOCCER.csv')
df.drop(columns=[
    'player_url', 'long_name', 'league_name', 'player_tags'
], inplace=True)

# 2) 기준 연도 및 가중치
base_year  = 2021
w_contract = 1.0
w_age      = 1.0
w_physic   = 1.2
w_pace     = 1.0
w_shooting = 1.0
w_stamina  = 0.8

# 3) 파생 변수
df['remaining_contract_years'] = df['contract_valid_until'] - base_year
df['age_factor'] = df['age'].max() - df['age']

# 4) 기대가치 계산
df['real_value_base'] = (
      w_contract * df['remaining_contract_years']
    + w_age      * df['age_factor']
    + w_physic   * df['physic']
    + w_pace     * df['pace']
    + w_shooting * df['shooting']
    + w_stamina  * df['power_stamina']
)
df['foot_bonus'] = df['preferred_foot'].map(lambda f: 1.1 if f=='Left' else 1.0)
df['real_value_expectation'] = df['real_value_base'] * df['foot_bonus']

# 5) NaN 제거
df = df[df['real_value_expectation'].notna()]

# 6) 랭크 생성
df['value_rank'] = (
    df['real_value_expectation']
      .rank(method='dense', ascending=False)
      .astype(int)
)

# 7) 결과 정리 & 저장
result = df[['value_rank', 'short_name', 'player_positions', 'real_value_expectation']] \
           .sort_values('value_rank')
result.to_csv('찬영/players_real_value_expectation.csv', index=False)