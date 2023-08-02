import pandas as pd




seasons=['2020','2021','2022','2023']

df = pd.DataFrame()
for season in seasons:
    subdf = pd.read_csv(f'date_stats/{season}_stats_opp_R.csv')

    df = pd.concat([df, pd.DataFrame(subdf, index=[0])])


print(df)