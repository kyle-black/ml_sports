import pandas as pd
import sqlite3
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

# Establish a connection to the database
connect_name = 'mlb_games.db'
connection = sqlite3.connect(connect_name)

# Load data from the database
past_games = pd.read_sql('SELECT * FROM past_games_update_5', connection)
odds = pd.read_sql('SELECT * FROM book_odds12', connection)

# Preprocess data
past_games['game_date'] = pd.to_datetime(past_games['game_date']).dt.date
odds = odds.groupby('game_id').apply(lambda group: group.fillna(method='ffill').fillna(method='bfill'))
odds['game_date'] = pd.to_datetime(odds['commence_time']).dt.tz_convert('US/Eastern').dt.date

# Merge dataframes
past_games = past_games.merge(odds, how='left', left_on=['home_team', 'away_team','game_date'], right_on=['home_team','away_team','game_date'])

# Select necessary columns
past_games = past_games[['id', 'game_date', 'season', 'home_team', 'away_team','home_is_winner', 'lowvig_home', 'lowvig_away', 'bovada_home', 'bovada_away', 'betonlineag_home', 'betonlineag_away', 'unibet_home', 'unibet_away', 'sport888_home', 'sport888_away', 'draftkings_home','draftkings_away','pinnacle_home', 'pinnacle_away', 'mybookieag_home', 'mybookieag_away']]
past_games =past_games.drop_duplicates(subset='id', keep='last')

# Function to convert American odds to implied probability
def american_to_implied_probability(american_odds):
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

# Convert odds to probabilities
for column in past_games.columns:
    if column.endswith('_home') or column.endswith('_away'):
        past_games[column] = past_games[column].apply(american_to_implied_probability)

# Remove rows where 'home_is_winner' is 'Unknown'
past_games = past_games[past_games['home_is_winner'] != 'Unknown']

# Drop rows with NaN values
past_games = past_games.dropna()

# Ensure home_is_winner is an integer
past_games['home_is_winner'] = past_games['home_is_winner'].astype(int)

# Compute Brier score for each column and create new columns
for column in past_games.columns:
    if column.endswith('_home') or column.endswith('_away'):
        # Compute Brier score
        score = brier_score_loss(past_games['home_is_winner'], past_games[column])
        
        # Create new column with Brier score
        past_games[column + '_brier'] = score

# Separate out home and away Brier scores
home_brier_scores = past_games.filter(like='_home_brier', axis=1)
away_brier_scores = past_games.filter(like='_away_brier', axis=1)

# Initialize a StandardScaler
scaler = StandardScaler()

# Apply Standard scaling to each Brier score in a row-wise manner for home
home_brier_scores_scaled = scaler.fit_transform(home_brier_scores.T).T
home_brier_scores_scaled = pd.DataFrame(home_brier_scores_scaled, columns=home_brier_scores.columns, index=home_brier_scores.index)

# Apply Standard scaling to each Brier score in a row-wise manner for away
away_brier_scores_scaled = scaler.fit_transform(away_brier_scores.T).T
away_brier_scores_scaled = pd.DataFrame(away_brier_scores_scaled, columns=away_brier_scores.columns, index=away_brier_scores.index)

# Combine the scaled home and away Brier scores back into the past_games DataFrame
past_games_scaled = pd.concat([past_games.drop(home_brier_scores.columns.tolist() + away_brier_scores.columns.tolist(), axis=1), home_brier_scores_scaled, away_brier_scores_scaled], axis=1)

# Save to CSV
past_games_scaled.to_csv('brier_score_df.csv')

print(past_games_scaled)