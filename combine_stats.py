import pandas as pd
import sqlite3
import numpy as np

conn = sqlite3.connect('mlb_games.db')



df =pd.read_csv('date_stats/winpct.csv')



df['Date']= pd.to_datetime(df['Date'])

df =df[['Date','Team', 'Current','Last 3','Last 1', 'Home', 'Away', 'Previous']]
#df.to_sql('win_pct', con=conn)


df_home = df.copy()

df_away = df.copy()
df_home.rename(columns ={'Date':'home_Date','Team':'home_Team', 'Current':'home_Current_win_pct', 'Last 3':'home_3_win_pct','Last 1':'home_1_win_pct','Home':'home_Home_win_pct', 'Away': 'home_Away_win_pct', 'Previous':'home_prev_win_pct'}, inplace =True)

teamname ={'Arizona':'Arizona Diamondbacks','Atlanta':'Atlanta Braves', 'Baltimore':'Baltimore Orioles','Boston':'Boston Red Sox','Chi Cubs':'Chicago Cubs','Chi White Sox':'Chicago White Sox','Cincinnati':'Cincinnati Reds','Cleveland':'Cleveland Guardians','Colorado':'Colorado Rockies','Detroit':'Detroit Tigers','Houston':'Houston Astros','Kansas City':'Kansas City Royals', 'LA Angels':'Los Angeles Angels', 'LA Dodgers': 'Los Angeles Dodgers','Miami':'Miami Marlins', 'Milwaukee':'Milwaukee Brewers','Minnesota':'Minnesota Twins','NY Mets':'New York Mets', 'NY Yankees':'New York Yankees', 'Oakland':'Oakland Athletics','Philadelphia':'Philadelphia Phillies','Pittsburgh':'Pittsburgh Pirates', 'San Diego': 'San Diego Padres','San Francisco': 'San Francisco Giants', 'Seattle':'Seattle Mariners', 'St. Louis':'St.Louis Cardinals', 'Tampa Bay':'Tampa Bay Rays','Texas':'Texas Rangers', 'Toronto': 'Toronto Blue Jays','Washington':'Washington Nationals'}
df_home['home_Team'].replace(teamname, inplace =True)
df_home.to_sql('home_win_pct', con=conn, if_exists='replace')



#df =df[['Date','Team', 'Current','Last 3','Last 1', 'Home', 'Away', 'Previous']]
#df.to_sql('win_pct', con=conn)



df_away.rename(columns ={'Date':'away_Date','Team':'away_Team', 'Current':'away_Current_win_pct', 'Last 3':'away_3_win_pct','Last 1':'away_1_win_pct','Home':'away_Home_win_pct', 'Away': 'away_Away_win_pct', 'Previous':'away_prev_win_pct'}, inplace=True)

teamname ={'Arizona':'Arizona Diamondbacks','Atlanta':'Atlanta Braves', 'Baltimore':'Baltimore Orioles','Boston':'Boston Red Sox','Chi Cubs':'Chicago Cubs','Chi White Sox':'Chicago Sox','Cincinnati':'Cincinnati Reds','Cleveland':'Cleveland Guardians','Colorado':'Colorado Rockies','Detroit':'Detroit Tigers','Houston':'Houston Astros','Kansas City':'Kansas City Royals', 'LA Angels':'Los Angeles Angels', 'LA Dodgers': 'Los Angeles Dodgers','Miami':'Miami Marlins', 'Milwaukee':'Milwaukee Brewers','Minnesota':'Minnesota Twins','NY Mets':'New York Mets', 'NY Yankees':'New York Yankees', 'Oakland':'Oakland Athletics','Philadelphia':'Philadelphia Phillies','Pittsburgh':'Pittsburgh Pirates', 'San Diego': 'San Diego Padres','SF Giants': 'San Francisco Giants', 'Seattle':'Seattle Mariners', 'St. Louis':'St.Louis Cardinals', 'Tampa Bay':'Tampa Bay Rays','Texas':'Texas Rangers', 'Toronto': 'Toronto Blue Jays','Washington':'Washington Nationals'}
df_away['away_Team'].replace(teamname, inplace =True)
df_away.to_sql('away_win_pct', con=conn, if_exists='replace')

df1 = pd.read_sql('SELECT * FROM past_games_update2', conn)
df1['game_date'] = pd.to_datetime(df1['game_date']).dt.normalize()
#print(df1['game_date'.to])

#df1['game_date'] = pd.to_datetime(df1['game_date'])

print(df1['game_date'])
#print(df_away['away_Date'])
merged_df = df1.merge(df_away, how='left', left_on=['away_team', 'game_date'], right_on = ['away_Team', 'away_Date'])


print(df_home.columns)

print(merged_df)
merged_df = merged_df.merge(df_home, how='left', left_on=['home_team', 'game_date'], right_on=['home_Team', 'home_Date'])
print(merged_df[300:340])
merged_df.dropna(inplace=True)


df_odds = pd.read_sql('SELECT * FROM book_odds12', conn)


print(df_odds.columns)

def find_home_cols(df):
    return [col for col in df if col.endswith('_home')]

def find_away_cols(df):
    return [col for col in df if col.endswith('_away')]


home_cols = find_home_cols(df_odds)
away_cols = find_away_cols(df_odds)




df_odds['median_home'] = df_odds[home_cols].apply(lambda row: row.median() if row.count() >= 5 else np.nan, axis=1)
df_odds['median_away'] = df_odds[away_cols].apply(lambda row: row.median() if row.count() >= 5 else np.nan, axis=1)


df_odds['mean_home'] = df_odds[home_cols].apply(lambda row: row.mean() if row.count() >= 5 else np.nan, axis=1)
df_odds['mean_away'] = df_odds[away_cols].apply(lambda row: row.mean() if row.count() >= 5 else np.nan, axis=1)

print(df_odds)




merged_df['game_date'] = merged_df['game_date'].dt.date

df_odds['commence_time'] = pd.to_datetime(df_odds['commence_time']).dt.tz_convert('US/Eastern').dt.date

print(df_odds['commence_time'])



#print(df_odds.columns)

print(df_odds.isna().sum())

merged_df =merged_df.merge(df_odds, how='left', left_on =['home_team','away_team','game_date'], right_on=['home_team', 'away_team', 'commence_time'])


print(merged_df.columns)

#merged_df = merged_df[merged_df.notna()]

merged_df =merged_df[['id', 'game_date', 'season', 'home_team', 'away_team', 'home_score',
       'away_score', 'home_is_winner', 'away_is_winner', 'game_type',
       'commence_time_x', 'away_Date', 'away_Current_win_pct',
       'away_3_win_pct', 'away_1_win_pct', 'away_Home_win_pct',
       'away_Away_win_pct', 'away_prev_win_pct', 'home_Date',
       'home_Current_win_pct', 'home_3_win_pct', 'home_1_win_pct',
       'home_Home_win_pct', 'home_Away_win_pct', 'home_prev_win_pct', 'median_home', 'median_away', 'mean_home', 'mean_away']]

#x =merged_df.isna().sum()
print(merged_df)

#merged_df.groupby('season')




merged_df =merged_df.drop_duplicates(subset='id', keep='last')


merged_df = merged_df.replace('--', np.nan).dropna()
merged_df = merged_df[merged_df.notna()]
print(merged_df)
#print(merged_df)
merged_df.to_sql('merged_df', conn, if_exists='replace')
merged_df.to_csv('updated_df1.csv')