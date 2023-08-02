import pandas as pd
import sqlite3

conn = sqlite3.connect('mlb_games.db')

#sql_query = '''
#SELECT * FROM book_odds12 JOIN past_games_update2 as pg 
#WHERE (book_odds12.home_team = pg.home_team) 
#AND (DATE(book_odds12.commence_time) = DATE(pg.commence_time)); '''



#sql_query='''
#SELECT * 
#FROM past_games_update2 AS PG 
#JOIN home_team_stats AS TS_home 
#    ON PG.home_team = TS_home.home_Tm 
#    AND PG.season = TS_home.home_season 
#JOIN away_team_stats AS TS_away 
#    ON PG.away_team = TS_away.away_Tm
#    AND PG.season = TS_away.away_season;
#'''


sql_query = '''
SELECT * 
FROM past_games_update2 AS PG 
JOIN home_team_stats AS TS_home 
    ON PG.home_team = TS_home.home_Tm 
    AND PG.season = TS_home.home_season 
JOIN away_team_stats AS TS_away 
    ON PG.away_team = TS_away.away_Tm
    AND PG.season = TS_away.away_season
LEFT JOIN book_odds12 as BO 
	ON BO.home_team = PG.home_team 
	AND (DATE(BO.commence_time) = DATE(pg.commence_time));
'''

df = pd.read_sql_query(sql_query, conn)

''''''
win_query ='''
SELECT * FROM win_pct;
'''

df_w= pd.read_sql_query(win_query, conn)

#print(df_w)
print(df)
#merged_df = df.merge(df_w, how='left', left_on=[])
'''
print(df)

#df.to_csv('df_games.csv')

# Drop rows where pinnacle_home or pinnacle_away are null
#df = df.dropna(subset=['pinnacle_home','pinnacle_away'])
#df = df.dropna(subset=[ 'pinnacle_home', 'pinnacle_away'])
df = df[df['home_score']!= 'Unknown']
df = df[df['away_score']!= 'Unknown']
#df = df.dropna(subset=['home_R','away_R','home_RA','away_RA' ,'home_is_winner'])
df = df.dropna(subset=['lowvig_home', 'lowvig_away'])
df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
df['home_diff'] = df['home_score']- df['away_score']
#df = df.drop_duplicates(subset='game_id', keep='last')
#df=df[['home_is_winner','home_score','away_score','home_diff','season','game_id','lowvig_home','pinnacle_home','pinnacle_away', 'lowvig_away','betonlineag_home','betonlineag_away','draftkings_home','draftkings_away','fanduel_away', 'fanduel_home']]


#df=df[['season','game_id','lowvig_home', 'lowvig_away','home_is_winner']]
#df=df[['season','game_id','home_team','away_team','pinnacle_home', 'pinnacle_away','lowvig_home', 'lowvig_away', 'home_score', 'away_score', 'home_is_winner']]
df =df[['game_id','commence_time','game_type','home_team','away_team','season','home_R','away_R','home_RA','away_RA' ,'home_is_winner', 'home_score','away_score', 'home_W-L%', 'away_W-L%','home_Luck', 'away_Luck','pinnacle_home', 'pinnacle_away', 'lowvig_home','lowvig_away', 'gtbets_home', 'gtbets_away']]


home = df[df['season'] == '2023'].groupby('home_team').agg(avg_runs_scored=('home_score', 'mean'))
away = df[df['season'] == '2023'].groupby('away_team').agg(avg_runs_scored=('away_score', 'mean'))

# merge and calculate average score
merged = pd.concat([home, away]).groupby(level=0).mean()
#print(df['pinnacle_home'])
#df =df[['season','home_team','away_team','game_id','pinnacle_home','pinnacle_away','home_is_winner']]
# Compute the median for home and away odds


#df['home_median'] = df.filter(like='_home').median()
#df['away_median'] = df.filter(like='_away').median()

# Fill null values with the corresponding median
#for col in df.columns:
#    if col.endswith('_home'):
#        df[col].fillna(medians_home[col], inplace=True)
#    elif col.endswith('_away'):
#        df[col].fillna(medians_away[col], inplace=True)
def american_to_implied_probability(american_odds):
    """
    Convert American odds to implied probability
    """
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)
    

def calculate_vig(american_odds1, american_odds2):
    """
    Calculate the vig given two American odds
    """

    # Calculate the implied probabilities from the odds
    prob1 = american_to_implied_probability(american_odds1)
    prob2 = american_to_implied_probability(american_odds2)

    # The vig is the excess of these probabilities over 1
    vig = prob1 + prob2 - 1

    return vig



#sportsbooks = ['pinnacle', 'lowvig', 'betonlineag', 'draftkings', 'fanduel']
sportsbooks =['lowvig']
# Loop through each sportsbook
for sportsbook in sportsbooks:
    home_col = f'{sportsbook}_home'
    away_col = f'{sportsbook}_away'

    # Convert American odds to implied probabilities
    df[f'{sportsbook}_home_prob'] = df.apply(lambda row: american_to_implied_probability(row[home_col]), axis=1)
    df[f'{sportsbook}_away_prob'] = df.apply(lambda row: american_to_implied_probability(row[away_col]), axis=1)

    # Calculate vig
    df[f'{sportsbook}_vig'] = df.apply(lambda row: calculate_vig(row[home_col], row[away_col]), axis=1)

    # Calculate vig free probabilities
    total_prob = df[f'{sportsbook}_home_prob'] + df[f'{sportsbook}_away_prob']
    df[f'{sportsbook}_home_vf'] = df[f'{sportsbook}_home_prob'] / total_prob
    df[f'{sportsbook}_away_vf'] = df[f'{sportsbook}_away_prob'] / total_prob

print(df)

df.to_csv('resultodds_new7.csv')
'''