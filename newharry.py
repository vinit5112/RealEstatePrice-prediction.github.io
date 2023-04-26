import pandas as pd
import numpy as np


df = pd.read_csv('hp1.csv')
# print(df.head())

df1 =  df[df['character'] == 'Harry Potter']
print(df1.head())

df1.to_csv('harry_rows.csv', index=False)

