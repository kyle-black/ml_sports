import sqlite3
import pandas as pd

def prepare_input_data(games):
    conn = sqlite3.connect('mlb_games.db')

    weighted_team_batting = pd.read_sql_query("SELECT * from weighted_team_batting", conn)
    weighted_team_fielding = pd.read_sql_query("SELECT * from weighted_team_fielding", conn)
    weighted_team_pitching = pd.read_sql_query("SELECT * from weighted_team_pitching", conn)

    conn.close()

    input_data = []
    for game in games:
        home_team = game['home_team']
        away_team = game['away_team']

        home_batting = weighted_team_batting.loc[weighted_team_batting['Tm'] == home_team].drop('Tm', axis=1)
        away_batting = weighted_team_batting.loc[weighted_team_batting['Tm'] == away_team].drop('Tm', axis=1)

        home_fielding = weighted_team_fielding.loc[weighted_team_fielding['Tm'] == home_team].drop('Tm', axis=1)
        away_fielding = weighted_team_fielding.loc[weighted_team_fielding['Tm'] == away_team].drop('Tm', axis=1)

        home_pitching = weighted_team_pitching.loc[weighted_team_pitching['Tm'] == home_team].drop('Tm', axis=1)
        away_pitching = weighted_team_pitching.loc[weighted_team_pitching['Tm'] == away_team].drop('Tm', axis=1)

        game_data = pd.concat([home_batting, away_batting, home_fielding, away_fielding, home_pitching, away_pitching], axis=1)
        input_data.append(game_data)

    input_data = pd.concat(input_data).reset_index(drop=True)


    



    
    
    #print(games.home_team)

    return input_data

# Example usage


games = [{'home_team': 'Arizona Diamondbacks', 'away_team': 'New York Yankees'},{'home_team': 'Arizona Diamondbacks', 'away_team': 'New York Yankees'} ]

print(prepare_input_data(games))




#input_data = prepare_input_data(games)
#print(input_data)