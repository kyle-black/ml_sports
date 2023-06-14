from datetime import datetime, timedelta
import requests
import time
import pandas as pd
import json
import sqlite3

conn = sqlite3.connect('mlb_games.db')

start_date = datetime(2020, 4, 1)
end_date = datetime(2023, 4, 30)
one_day = timedelta(days=1)

date_list = []

current_date = start_date
while current_date <= end_date:
    date_string = current_date.strftime('%Y-%m-%dT12:00:00Z')
    date_list.append(date_string)
    current_date += one_day

print(date_list)

request_dict = {}
for i in date_list[:]:
    req = requests.get(
        f'https://api.the-odds-api.com/v4/sports/baseball_mlb/odds-history/?apiKey=c47c3bdfd870b70a41d35c839dcab514&regions=us&markets=h2h&oddsFormat=american&date={i}')

    request_dict[i] = req.text
    time.sleep(1)

print(request_dict)

rows = []
columns = ["game_id", "home_team", "away_team", "commence_time"]
df = pd.DataFrame(columns=columns)

for date_key in request_dict:
    data = json.loads(request_dict[date_key])

    games_data = data["data"]

    for game_data in games_data:
        game_id = game_data["id"]
        home_team = game_data["home_team"]
        away_team = game_data["away_team"]
        commence_time = game_data["commence_time"]

        row_data = {
            "game_id": game_id,
            "home_team": home_team,
            "away_team": away_team,
            "commence_time": commence_time,
        }

        for bookmaker in game_data["bookmakers"]:
            key = bookmaker["key"]

            outcomes = bookmaker["markets"][0]["outcomes"]

            if outcomes[0]["name"] == home_team:
                home_odds = outcomes[0]["price"]
                away_odds = outcomes[1]["price"]
            else:
                home_odds = outcomes[1]["price"]
                away_odds = outcomes[0]["price"]

            row_data[key + "_home"] = home_odds
            row_data[key + "_away"] = away_odds

        rows.append(row_data)

df = pd.DataFrame(rows)

print(df)
df.to_csv('bookmakers.csv')
conn = sqlite3.connect("mlb_games.db")

df.to_sql("book_odds9", conn, if_exists="replace", index=False)

conn.close()