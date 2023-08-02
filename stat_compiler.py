import pandas as pd
import sqlite3

conn = sqlite3.connect('mlb_games.db')
years = ['2019','2020','2021','2022','2023']

df = pd.DataFrame()

home_df = pd.DataFrame()
away_df = pd.DataFrame()

for year in years:
    df_home = pd.read_csv(f'mlb_stats/{year}/{year}_team.csv')
    df_home['season'] = year
    # rename columns for the home_df
    df_home = df_home.rename(columns={col: 'home_' + col for col in df_home.columns})
    
    df_away = pd.read_csv(f'mlb_stats/{year}/{year}_team.csv')
    df_away['season'] = year
    # rename columns for the away_df
    df_away = df_away.rename(columns={col: 'away_' + col for col in df_away.columns})

    home_df = pd.concat([home_df,df_home])
    away_df = pd.concat([away_df,df_away])

# print one of the dataframes to verify
print(home_df)

home_df.to_sql('home_team_stats', conn, if_exists='replace', index=False)
away_df.to_sql('away_team_stats', conn, if_exists='replace', index=False)