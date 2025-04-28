import pandas as pd

df = pd.read_csv('csvfile/5_SOCCER.csv')
df.drop(columns=[
    'player_url', 'long_name', 'league_name', 'player_tags'
], inplace=True)

base_year  = 2021
w_contract = 1.0
w_age      = 1.0
w_physic   = 1.2
w_pace     = 1.0
w_shooting = 1.0
w_stamina  = 0.8

df['remaining_contract_years'] = df['contract_valid_until'] - base_year
df['age_factor'] = df['age'].max() - df['age']

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

df = df[df['real_value_expectation'].notna()]

df['value_rank'] = (
    df['real_value_expectation']
      .rank(method='dense', ascending=False)
      .astype(int)
)

result = df[['value_rank', 'short_name', 'player_positions', 'real_value_expectation']] \
           .sort_values('value_rank')
result.to_csv('찬영/players_real_value_expectation.csv', index=False)