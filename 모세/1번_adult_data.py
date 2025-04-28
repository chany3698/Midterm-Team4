import pandas as pd        
import numpy as np                     
import matplotlib.pyplot as plt              
import seaborn as sns                                
from sklearn.preprocessing import LabelEncoder, MinMaxScaler                   

df=pd.read_csv(r"E:/Midterm-Team4/모세/1_adults.csv")                     
print(df.head())                  
print(df.isnull().sum())               


le = LabelEncoder()             
df['sex'] = le.fit_transform(df['sex'])              

numeric_cols = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']              

for col in numeric_cols:                                                      
    df[col] = pd.to_numeric(df[col], errors='coerce')                        


missing_like = ['ERROR', 'UNKNOWN', 'missing', 'Missing', 'N/A', 'na', 'NaN', "?"]                                      

df.replace(missing_like, pd.NA, inplace=True)                                       


for col in df.columns:                                                            
    if df[col].dtype == 'object':                                      
        df[col] = df[col].fillna('Unknown')                            
    else:                                                              
        df[col] = df[col].fillna(df[col].mean())                       


scaler = MinMaxScaler()                                                                                   
df[['age']] = scaler.fit_transform(df[['age']])                     

Q1 = df['age'].quantile(0.25)                                     
Q3 = df['age'].quantile(0.75)                                     
IQR = Q3 - Q1


condition = (df['age'] >= (Q1 - 1.5 * IQR)) & (df['age'] <= (Q3 + 1.5 * IQR))              
df = df[condition]

df.drop_duplicates(inplace=True)

df['income_level'] = df['income'].apply(lambda x: '$50,000 미만' if x == '<=50K' else '$50,000 이상')                                         

condition2 = (df['income_level']=="$50,000 이상")              
df = df[condition2]

print(df.isnull().sum())                                                       
print(df.head())                                  


output_path = 'adults_final_data.csv'                              
df.to_csv(output_path, index=False, encoding='utf-8-sig')                  