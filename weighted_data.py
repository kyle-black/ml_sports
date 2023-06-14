import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect('mlb_games.db')

# Read the tables from the database
team_batting = pd.read_sql_query("SELECT * from team_batting", conn)
team_fielding = pd.read_sql_query("SELECT * from team_fielding", conn)
team_pitching = pd.read_sql_query("SELECT * from team_pitching", conn)

# Close the database connection
conn.close()

# Set the year column name
year_column = 'season'

# Filter dataframes for 2022 and 2023, drop the 'season' column, and set 'Tm' as index
team_batting_2022 = team_batting[team_batting[year_column] == 2022].drop('season', axis=1).set_index('Tm')
team_fielding_2022 = team_fielding[team_fielding[year_column] == 2022].drop('season', axis=1).set_index('Tm')
team_pitching_2022 = team_pitching[team_pitching[year_column] == 2022].drop('season', axis=1).set_index('Tm')

team_batting_2023 = team_batting[team_batting[year_column] == 2023].drop('season', axis=1).set_index('Tm')
team_fielding_2023 = team_fielding[team_fielding[year_column] == 2023].drop('season', axis=1).set_index('Tm')
team_pitching_2023 = team_pitching[team_pitching[year_column] == 2023].drop('season', axis=1).set_index('Tm')

# Number of games played in 2022 and 2023
games_played_2022 = 162
games_played_2023 = 30

# Calculate the weights for 2022 and 2023
total_games = games_played_2022 + games_played_2023
weight_2022 = games_played_2022 / total_games
weight_2023 = games_played_2023 / total_games

# Calculate the weighted dataframes for each category
weighted_team_batting = weight_2022 * team_batting_2022 + weight_2023 * team_batting_2023
weighted_team_fielding = weight_2022 * team_fielding_2022 + weight_2023 * team_fielding_2023
weighted_team_pitching = weight_2022 * team_pitching_2022 + weight_2023 * team_pitching_2023

# Reset the index and drop the index name
weighted_team_batting.reset_index(inplace=True)
weighted_team_fielding.reset_index(inplace=True)
weighted_team_pitching.reset_index(inplace=True)

weighted_team_batting.index.name = None
weighted_team_fielding.index.name = None
weighted_team_pitching.index.name = None

# Update the SQLite database with the new weighted dataframes
conn = sqlite3.connect('mlb_games.db')

weighted_team_batting.to_sql('weighted_team_batting', conn, if_exists='replace', index=False)
weighted_team_fielding.to_sql('weighted_team_fielding', conn, if_exists='replace', index=False)
weighted_team_pitching.to_sql('weighted_team_pitching', conn, if_exists='replace', index=False)

conn.close()

