from datetime import datetime, timedelta
import requests
import time
import pandas as pd
import json
import sqlite3
import numpy as np

conn = sqlite3.connect('mlb_games.db')

def odds_pull_sequence():
    start_date = datetime(2023, 5, 1)
    end_date = datetime(2023, 7, 1)
    one_day = timedelta(days=1)

    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_string = current_date.strftime('%Y-%m-%dT12:00:00Z')
        date_list.append(date_string)
        current_date += one_day

    request_dict = {}
    for i in date_list[:]:
        req = requests.get(
            f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds-history/?apiKey=1c32d390b73f33d8e39c74c53d373c45&regions=us,eu&markets=h2h&oddsFormat=american&date={i}')
        print(req.text)
        request_dict[i] = req.text
        time.sleep(2)

    # List of all bookmaker columns (home and away) that you expect to be present
    bookmaker_columns = ['lowvig_home', 'lowvig_away',
        'betonlineag_home', 'betonlineag_away', 'unibet_home', 'unibet_away',
        'sport888_home', 'sport888_away', 'draftkings_home', 'draftkings_away',
        'pointsbetus_home', 'pointsbetus_away', 'gtbets_home', 'gtbets_away',
        'mybookieag_home', 'mybookieag_away', 'bovada_home', 'bovada_away',
        'williamhill_home', 'williamhill_away', 'onexbet_home', 'onexbet_away',
        'marathonbet_home', 'marathonbet_away', 'nordicbet_home',
        'nordicbet_away', 'fanduel_home', 'fanduel_away', 'betfair_home',
        'betfair_away', 'matchbook_home', 'matchbook_away', 'intertops_home',
        'intertops_away', 'pinnacle_home', 'pinnacle_away',
        'williamhill_us_home', 'williamhill_us_away', 'betrivers_home',
        'betrivers_away', 'betmgm_home', 'betmgm_away', 'betclic_home',
        'betclic_away', 'caesars_home', 'caesars_away', 'sugarhouse_home',
        'sugarhouse_away', 'betway_home', 'betway_away', 'foxbet_home',
        'foxbet_away', 'barstool_home', 'barstool_away', 'twinspires_home',
        'twinspires_away', 'betus_home', 'betus_away', 'wynnbet_home',
        'wynnbet_away', 'circasports_home', 'circasports_away',
        'superbook_home', 'superbook_away', 'unibet_us_home', 'unibet_us_away',
        'unibet_eu_home', 'unibet_eu_away']

    rows = []

    for date_key in request_dict:
        data = json.loads(request_dict[date_key])
        games_data = data["data"]

        for game_data in games_data:
            # Initiate all bookmaker columns with np.nan
            row_data = {bookmaker: np.nan for bookmaker in bookmaker_columns}
            # Then you can add your common columns
            row_data.update({
                "game_id": game_data["id"],
                "home_team": game_data["home_team"],
                "away_team": game_data["away_team"],
                "commence_time": game_data["commence_time"],
            })

            for bookmaker in game_data["bookmakers"]:
                key = bookmaker["key"]

                outcomes = bookmaker["markets"][0]["outcomes"]

                if outcomes[0]["name"] == game_data["home_team"]:
                    home_odds = outcomes[0]["price"]
                    away_odds = outcomes[1]["price"]
                else:
                    home_odds = outcomes[1]["price"]
                    away_odds = outcomes[0]["price"]

                row_data[key + "_home"] = home_odds
                row_data[key + "_away"] = away_odds

            rows.append(row_data)

    df = pd.DataFrame(rows)
    df = df.drop(['betfair_ex_eu_home','betfair_ex_eu_away'], axis=1)
    print(df)
    df.to_csv('bookmakers.csv')
    conn = sqlite3.connect("mlb_games.db")

    df.to_sql("book_odds12", conn, if_exists="append", index=False)

    conn.close()