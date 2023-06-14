import sqlite3
import pandas as pd

def get_weighted_data_for_games(database_name):
    conn = sqlite3.connect(database_name)
    cursor = conn.cursor()

    # Create an empty list to store the results
    result_data = []

    games = [{'home_team': 'Arizona Diamondbacks', 'away_team': 'New York Yankees'}, {'home_team': 'Arizona Diamondbacks', 'away_team': 'New York Yankees'}]

    for game in games:
        home_team = game['home_team']
        away_team = game['away_team']

        # Fetch home team weighted stats
        cursor.execute(f"SELECT * FROM weighted_team_batting WHERE Tm = '{home_team}';")
        home_weighted_batting = cursor.fetchone() or (0,) * 29
        cursor.execute(f"SELECT * FROM weighted_team_fielding WHERE Tm = '{home_team}';")
        home_weighted_fielding = cursor.fetchone() or (0,) * 19
        cursor.execute(f"SELECT * FROM weighted_team_pitching WHERE Tm = '{home_team}';")
        home_weighted_pitching = cursor.fetchone() or (0,) * 31

        # Fetch away team weighted stats
        cursor.execute(f"SELECT * FROM weighted_team_batting WHERE Tm = '{away_team}';")
        away_weighted_batting = cursor.fetchone() or (0,) * 29
        cursor.execute(f"SELECT * FROM weighted_team_fielding WHERE Tm = '{away_team}';")
        away_weighted_fielding = cursor.fetchone() or (0,) * 19
        cursor.execute(f"SELECT * FROM weighted_team_pitching WHERE Tm = '{away_team}';")
        away_weighted_pitching = cursor.fetchone() or (0,) * 31

        # Combine fetched data into a single row
        row_data = (home_team, away_team) + home_weighted_batting + home_weighted_fielding + home_weighted_pitching + away_weighted_batting + away_weighted_fielding + away_weighted_pitching
        result_data.append(row_data)

    conn.close()

    # Create a DataFrame from the result_data list
    result_df = pd.DataFrame(result_data)

    # Print the resulting DataFrame (you can modify this part to process the data as needed)
    print(result_df.columns)

database_name = 'mlb_games.db'  # Replace with the name of your SQLite database
get_weighted_data_for_games(database_name)