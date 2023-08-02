import pandas as pd
import sqlite3
import numpy as np

conn = sqlite3.connect('mlb_games.db')


# Load your existing data
df = pd.read_sql('SELECT * FROM past_games_update_4', conn)

# Convert the 'game_date' column to a datetime object
df['game_date'] = pd.to_datetime(df['game_date'])

# Filter the DataFrame where game_type equals 'R'
df = df[df['game_type'] == 'R']

# Create a temporary DataFrame for home games
df_home = df[['game_date', 'season', 'home_team']].copy() # Copy the DataFrame
df_home['team'] = df_home['home_team']

# Create a temporary DataFrame for away games
df_away = df[['game_date', 'season', 'away_team']].copy() # Copy the DataFrame
df_away['team'] = df_away['away_team']

# Concatenate home and away DataFrames
df_total = pd.concat([df_home, df_away])

# Sort by date and season, and reset the index for cumcount
df_total = df_total.sort_values(['season', 'team', 'game_date']).reset_index(drop=True)

# Add a 'game_counter' column that is a cumulative count of games for each team in each season
df_total['game_counter'] = df_total.groupby(['season', 'team']).cumcount() + 1

# Merge the 'game_counter' back into the original DataFrame for home team
df = df.merge(df_total[['game_date', 'season', 'team', 'game_counter']], 
              how='left', 
              left_on=['game_date', 'season', 'home_team'], 
              right_on=['game_date', 'season', 'team'])

# Drop the redundant 'team' column and rename 'game_counter'
df = df.drop(columns='team')
df = df.rename(columns={'game_counter': 'home_team_game_counter'})

# Merge the 'game_counter' back into the original DataFrame for away team
df = df.merge(df_total[['game_date', 'season', 'team', 'game_counter']], 
              how='left', 
              left_on=['game_date', 'season', 'away_team'], 
              right_on=['game_date', 'season', 'team'])

# Drop the redundant 'team' column and rename 'game_counter'
df = df.drop(columns='team')
df = df.rename(columns={'game_counter': 'away_team_game_counter'})

# Now 'df' should have the cumulative count of games played by home team and away team up to each game
df.to_sql('past_games_update_5', conn, if_exists='replace')