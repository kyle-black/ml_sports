import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

df =pd.read_csv('resultodds_new6.csv')

df = df.dropna(subset=['home_is_winner'])
df = df.dropna(subset=['lowvig_home'])
df = df[df['home_is_winner'] != 'Unknown']
#r2 = r2_score(df['lowvig_home'], df['home_is_winner'])

print(df.describe())
