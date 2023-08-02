import pandas as pd
import numpy as np


df = pd.read_csv('updated_df1.csv')

#df['game_date'] = pd.to_datetime(df['game_date']).dt.date

#print(df['game_date'])
df['game_date'] = pd.to_datetime(df['game_date'])

df_ = pd.read_csv('date_stats/stats_R.csv')
#print(df_h)
df_h = df_.copy()
df_a = df_.copy()

#home team df
teamname ={'Arizona':'Arizona Diamondbacks','Atlanta':'Atlanta Braves', 'Baltimore':'Baltimore Orioles','Boston':'Boston Red Sox','Chi Cubs':'Chicago Cubs','Chi White Sox':'Chicago Sox','Cincinnati':'Cincinnati Reds','Cleveland':'Cleveland Guardians','Colorado':'Colorado Rockies','Detroit':'Detroit Tigers','Houston':'Houston Astros','Kansas City':'Kansas City Royals', 'LA Angels':'Los Angeles Angels', 'LA Dodgers': 'Los Angeles Dodgers','Miami':'Miami Marlins', 'Milwaukee':'Milwaukee Brewers','Minnesota':'Minnesota Twins','NY Mets':'New York Mets', 'NY Yankees':'New York Yankees', 'Oakland':'Oakland Athletics','Philadelphia':'Philadelphia Phillies','Pittsburgh':'Pittsburgh Pirates', 'San Diego': 'San Diego Padres','SF Giants': 'San Francisco Giants', 'Seattle':'Seattle Mariners', 'St. Louis':'St.Louis Cardinals', 'Tampa Bay':'Tampa Bay Rays','Texas':'Texas Rangers', 'Toronto': 'Toronto Blue Jays','Washington':'Washington Nationals'}
df_h['Team'].replace(teamname, inplace =True)

print(df_h.columns)
df_h =df_h.rename(columns={'Date':'home_r_Date','Team':'home_r_Team','Current':'home_r_Current','Last 3':'home_r_3','Last 1':'home_r_1','Home':'home_r_Home','Previous':'home_r_Previous'})

#print(df)


df_h =df_h[['home_r_Date', 'home_r_Team','home_r_Current','home_r_3', 'home_r_1','home_r_Home', 'home_r_Previous']]
df_h = df_h[df_h['home_r_Date'] != 'Date']
df_h['home_r_Date'] = pd.to_datetime(df_h['home_r_Date'])

print(df_h['home_r_Date'])

df_merged= df.merge(df_h, how='left', left_on=['home_team','game_date'], right_on=['home_r_Team', 'home_r_Date'])


# away team df 

teamname ={'Arizona':'Arizona Diamondbacks','Atlanta':'Atlanta Braves', 'Baltimore':'Baltimore Orioles','Boston':'Boston Red Sox','Chi Cubs':'Chicago Cubs','Chi White Sox':'Chicago Sox','Cincinnati':'Cincinnati Reds','Cleveland':'Cleveland Guardians','Colorado':'Colorado Rockies','Detroit':'Detroit Tigers','Houston':'Houston Astros','Kansas City':'Kansas City Royals', 'LA Angels':'Los Angeles Angels', 'LA Dodgers': 'Los Angeles Dodgers','Miami':'Miami Marlins', 'Milwaukee':'Milwaukee Brewers','Minnesota':'Minnesota Twins','NY Mets':'New York Mets', 'NY Yankees':'New York Yankees', 'Oakland':'Oakland Athletics','Philadelphia':'Philadelphia Phillies','Pittsburgh':'Pittsburgh Pirates', 'San Diego': 'San Diego Padres','SF Giants': 'San Francisco Giants', 'Seattle':'Seattle Mariners', 'St. Louis':'St.Louis Cardinals', 'Tampa Bay':'Tampa Bay Rays','Texas':'Texas Rangers', 'Toronto': 'Toronto Blue Jays','Washington':'Washington Nationals'}
df_a['Team'].replace(teamname, inplace =True)

print(df_a.columns)

df_a =df_a.rename(columns={'Date':'away_r_Date','Team':'away_r_Team','Current':'away_r_Current','Last 3':'away_r_3','Last 1':'away_r_1','Home':'away_r_Home','Previous':'away_r_Previous'})

#print(df)


df_a =df_a[['away_r_Date', 'away_r_Team','away_r_Current','away_r_3', 'away_r_1','away_r_Home', 'away_r_Previous']]
df_a = df_a[df_a['away_r_Date'] != 'Date']
df_a['away_r_Date'] = pd.to_datetime(df_a['away_r_Date'])

#df_a['away_r_Date'] = pd.to_datetime(df_a['away_r_Date'])
df_merged= df_merged.merge(df_a, how='left', left_on=['away_team', 'game_date'], right_on=['away_r_Team','away_r_Date'])


print(df_merged)
print(df_merged.columns)
df_merged =df_merged.drop_duplicates(subset='id', keep='last')

print(df_merged.columns)
df_merged.to_csv('updated_df2.csv')
#df_merged.to_csv('updated_df2.csv')
#df_merged = df_merged.merge(df_r, how='left', left_on=['away_team','game_date'], right_on=['Team','Date'])


'''
df_opp = pd.read_csv('date_stats/stats_opp_R.csv')
df_h_opp = df_opp.copy()
df_a_opp = df_opp.copy()
#home team df
teamname ={'Arizona':'Arizona Diamondbacks','Atlanta':'Atlanta Braves', 'Baltimore':'Baltimore Orioles','Boston':'Boston Red Sox','Chi Cubs':'Chicago Cubs','Chi White Sox':'Chicago Sox','Cincinnati':'Cincinnati Reds','Cleveland':'Cleveland Guardians','Colorado':'Colorado Rockies','Detroit':'Detroit Tigers','Houston':'Houston Astros','Kansas City':'Kansas City Royals', 'LA Angels':'Los Angeles Angels', 'LA Dodgers': 'Los Angeles Dodgers','Miami':'Miami Marlins', 'Milwaukee':'Milwaukee Brewers','Minnesota':'Minnesota Twins','NY Mets':'New York Mets', 'NY Yankees':'New York Yankees', 'Oakland':'Oakland Athletics','Philadelphia':'Philadelphia Phillies','Pittsburgh':'Pittsburgh Pirates', 'San Diego': 'San Diego Padres','SF Giants': 'San Francisco Giants', 'Seattle':'Seattle Mariners', 'St. Louis':'St.Louis Cardinals', 'Tampa Bay':'Tampa Bay Rays','Texas':'Texas Rangers', 'Toronto': 'Toronto Blue Jays','Washington':'Washington Nationals'}
df_h_opp['Team'].replace(teamname, inplace =True)


print(df_h.columns)
df_h_opp =df_h_opp.rename(columns={'Date':'home_opp_Date','Team':'home_opp_Team','Current':'home_opp_Current','Last 3':'home_opp_3','Last 1':'home_opp_1','Home':'home_opp_Home','Previous':'home_opp_Previous'})
df_h_opp['home_opp_Date'] = pd.to_datetime(df_h_opp['home_opp_Date'])
df_merged= df_merged.merge(df_h_opp, how='left', left_on=['home_team','game_date'], right_on=['home_opp_Team', 'home_opp_Date'])


teamname ={'Arizona':'Arizona Diamondbacks','Atlanta':'Atlanta Braves', 'Baltimore':'Baltimore Orioles','Boston':'Boston Red Sox','Chi Cubs':'Chicago Cubs','Chi White Sox':'Chicago Sox','Cincinnati':'Cincinnati Reds','Cleveland':'Cleveland Guardians','Colorado':'Colorado Rockies','Detroit':'Detroit Tigers','Houston':'Houston Astros','Kansas City':'Kansas City Royals', 'LA Angels':'Los Angeles Angels', 'LA Dodgers': 'Los Angeles Dodgers','Miami':'Miami Marlins', 'Milwaukee':'Milwaukee Brewers','Minnesota':'Minnesota Twins','NY Mets':'New York Mets', 'NY Yankees':'New York Yankees', 'Oakland':'Oakland Athletics','Philadelphia':'Philadelphia Phillies','Pittsburgh':'Pittsburgh Pirates', 'San Diego': 'San Diego Padres','SF Giants': 'San Francisco Giants', 'Seattle':'Seattle Mariners', 'St. Louis':'St.Louis Cardinals', 'Tampa Bay':'Tampa Bay Rays','Texas':'Texas Rangers', 'Toronto': 'Toronto Blue Jays','Washington':'Washington Nationals'}
df_a_opp['Team'].replace(teamname, inplace =True)

df_a_opp =df_a_opp.rename(columns={'Date':'away_opp_Date','Team':'away_opp_Team','Current':'away_opp_Current','Last 3':'away_opp_3','Last 1':'away_opp_1','Home':'away_opp_Home','Previous':'away_opp_Previous'})
df_a_opp['away_opp_Date'] = pd.to_datetime(df_a_opp['away_opp_Date'])
df_merged= df_merged.merge(df_a_opp, how='left', left_on=['away_team','game_date'], right_on=['away_opp_Team', 'away_opp_Date'])
df_merged = df_merged.replace('--', np.nan).dropna()

print(df_merged)


#print(df['game_date'])



#print(df['home_Date'])
'''