import pandas as pd


df = pd.read_csv('date_stats/winpct.csv')


print(df.columns)

#df = df.drop(columns='id')
df_reset = df.reset_index()
#print(df.reset)
df_reset['id'] = df_reset.index

df_reset.to_csv('date_stats/winpct_r.csv')
