import pandas as pd
import numpy as np
import sqlite3
import numpy as np

conn = sqlite3.connect('mlb_games.db')


df_merged = pd.read_csv('updated_df2.csv')

df_merged['game_date'] = pd.to_datetime(df_merged['game_date'])

df_opp = pd.read_csv('date_stats/stats_opp_R.csv')
df_h_opp = df_opp.copy()
df_a_opp = df_opp.copy()
#home team df
teamname ={'Arizona':'Arizona Diamondbacks','Atlanta':'Atlanta Braves', 'Baltimore':'Baltimore Orioles','Boston':'Boston Red Sox','Chi Cubs':'Chicago Cubs','Chi White Sox':'Chicago Sox','Cincinnati':'Cincinnati Reds','Cleveland':'Cleveland Guardians','Colorado':'Colorado Rockies','Detroit':'Detroit Tigers','Houston':'Houston Astros','Kansas City':'Kansas City Royals', 'LA Angels':'Los Angeles Angels', 'LA Dodgers': 'Los Angeles Dodgers','Miami':'Miami Marlins', 'Milwaukee':'Milwaukee Brewers','Minnesota':'Minnesota Twins','NY Mets':'New York Mets', 'NY Yankees':'New York Yankees', 'Oakland':'Oakland Athletics','Philadelphia':'Philadelphia Phillies','Pittsburgh':'Pittsburgh Pirates', 'San Diego': 'San Diego Padres','SF Giants': 'San Francisco Giants', 'Seattle':'Seattle Mariners', 'St. Louis':'St.Louis Cardinals', 'Tampa Bay':'Tampa Bay Rays','Texas':'Texas Rangers', 'Toronto': 'Toronto Blue Jays','Washington':'Washington Nationals'}
df_h_opp['Team'].replace(teamname, inplace =True)


#print(df_h.columns)
df_h_opp =df_h_opp.rename(columns={'Date':'home_opp_Date','Team':'home_opp_Team','Current':'home_opp_Current','Last 3':'home_opp_3','Last 1':'home_opp_1','Home':'home_opp_Home','Previous':'home_opp_Previous'})
df_h_opp['home_opp_Date'] = pd.to_datetime(df_h_opp['home_opp_Date'])
df_merged= df_merged.merge(df_h_opp, how='left', left_on=['home_team','game_date'], right_on=['home_opp_Team', 'home_opp_Date'])

#print(df_merged)


teamname ={'Arizona':'Arizona Diamondbacks','Atlanta':'Atlanta Braves', 'Baltimore':'Baltimore Orioles','Boston':'Boston Red Sox','Chi Cubs':'Chicago Cubs','Chi White Sox':'Chicago Sox','Cincinnati':'Cincinnati Reds','Cleveland':'Cleveland Guardians','Colorado':'Colorado Rockies','Detroit':'Detroit Tigers','Houston':'Houston Astros','Kansas City':'Kansas City Royals', 'LA Angels':'Los Angeles Angels', 'LA Dodgers': 'Los Angeles Dodgers','Miami':'Miami Marlins', 'Milwaukee':'Milwaukee Brewers','Minnesota':'Minnesota Twins','NY Mets':'New York Mets', 'NY Yankees':'New York Yankees', 'Oakland':'Oakland Athletics','Philadelphia':'Philadelphia Phillies','Pittsburgh':'Pittsburgh Pirates', 'San Diego': 'San Diego Padres','SF Giants': 'San Francisco Giants', 'Seattle':'Seattle Mariners', 'St. Louis':'St.Louis Cardinals', 'Tampa Bay':'Tampa Bay Rays','Texas':'Texas Rangers', 'Toronto': 'Toronto Blue Jays','Washington':'Washington Nationals'}
df_a_opp['Team'].replace(teamname, inplace =True)



df_a_opp =df_a_opp.rename(columns={'Date':'away_opp_Date','Team':'away_opp_Team','Current':'away_opp_Current','Last 3':'away_opp_3','Last 1':'away_opp_1','Home':'away_opp_Home','Previous':'away_opp_Previous'})
df_a_opp['away_opp_Date'] = pd.to_datetime(df_a_opp['away_opp_Date'])

print(df_merged)
print(df_a_opp)


df_merged= df_merged.merge(df_a_opp, how='left', left_on=['away_team','game_date'], right_on=['away_opp_Team', 'away_opp_Date'])
print(df_merged.columns)




df_merged= df_merged[['id', 'game_date', 'season',
       'home_team', 'away_team', 'home_score', 'away_score', 'home_is_winner',
       'away_is_winner', 'game_type', 'away_Date', 
       'away_Current_win_pct', 'away_3_win_pct', 'away_1_win_pct',
       'away_Home_win_pct', 'away_Away_win_pct', 'away_prev_win_pct',
       'home_Date', 'home_Current_win_pct', 'home_3_win_pct',
       'home_1_win_pct', 'home_Home_win_pct', 'home_Away_win_pct',
       'home_prev_win_pct', 'home_r_Date', 'home_r_Team', 'home_r_Current',
       'home_r_3', 'home_r_1', 'home_r_Home', 'home_r_Previous', 'away_r_Date',
       'away_r_Team', 'away_r_Current', 'away_r_3', 'away_r_1', 'away_r_Home',
       'away_r_Previous','home_opp_Date', 'home_opp_Team',
       'home_opp_Current', 'home_opp_3', 'home_opp_1', 'home_opp_Home',
        'home_opp_Previous','away_opp_Date', 'away_opp_Team', 'away_opp_Current', 'away_opp_3',
       'away_opp_1', 'away_opp_Home', 'away_opp_Previous', 'home_median', 'away_median', 'home_mean', 'away_mean']]
'''
df_merged =df_merged[['id', 'game_date', 'season',
       'home_team', 'away_team', 'home_score', 'away_score', 'home_is_winner',
       'away_is_winner', 'game_type', 'commence_time_x', 'away_Date',
       'away_Team', 'away_Current_win_pct', 'away_3_win_pct', 'away_1_win_pct',
       'away_Home_win_pct', 'away_Away_win_pct', 'away_prev_win_pct',
       'home_Date', 'home_Team', 'home_Current_win_pct', 'home_3_win_pct',
       'home_1_win_pct', 'home_Home_win_pct', 'home_Away_win_pct',
       'home_prev_win_pct', 'lowvig_home', 'lowvig_away', 'home_r_Date',
       'home_r_Team', 'home_r_Current', 'home_r_3', 'home_r_1', 'home_r_Home',
       'home_r_Previous', 'away_r_Date', 'away_r_Team', 'away_r_Current',
       'away_r_3', 'away_r_1', 'away_r_Home', 'away_r_Previous',
        'home_opp_Date', 'home_opp_Team', 'home_opp_Current',
       'home_opp_3', 'home_opp_1', 'home_opp_Home', 'Away_x',
       'home_opp_Previous','away_opp_Date',
       'away_opp_Team', 'away_opp_Current', 'away_opp_3', 'away_opp_1',
       'away_opp_Home', 'Away_y', 'away_opp_Previous']]

print(df_merged)

'''
df_merged = df_merged.replace('--', np.nan).dropna()


df_merged.to_csv('df_final.csv')

df_merged.to_sql('df_final3', conn)
print(df_merged)
