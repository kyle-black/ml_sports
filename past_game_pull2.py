import sqlite3
import pandas as pd
import json

# Connect to the SQLite database
conn = sqlite3.connect('mlb_games.db')

# Define the SQL query
sql_query = '''
SELECT * FROM book_odds5 JOIN past_games WHERE (book_odds5.home_team = past_games.home_team) AND (book_odds5.commence_time = past_games.commence_time);
'''

# Execute the query and fetch the results as a DataFrame
df = pd.read_sql_query(sql_query, conn)




odds_columns = ['lowvig', 'betonlineag', 'unibet', 'draftkings', 'pointsbetus', 'gtbets', 'mybookieag', 'bovada', 'fanduel', 'betfair', 'intertops', 'williamhill_us', 'betrivers', 'betmgm', 'caesars', 'sugarhouse', 'foxbet', 'barstool', 'twinspires', 'betus', 'wynnbet', 'circasports', 'superbook', 'unibet_us']
#df[odds_columns] = df[odds_columns].applymap(parse_odds)

print(df['lowvig'])
'''
df = df[df[odds_columns].isnull().sum(axis=1) <= 10]

#print(df)


for col in odds_columns:
    df[f"{col}_away"] = df[col].apply(lambda x: x[0] if x is not None else None)
    df[f"{col}_home"] = df[col].apply(lambda x: x[1] if x is not None else None)

# Drop original odds columns
df = df.drop(columns=odds_columns)

# Calculate the median of away columns and home columns
away_columns = [f"{col}_away" for col in odds_columns]
home_columns = [f"{col}_home" for col in odds_columns]

df["away_median"] = df[away_columns].median(axis=1)
df["home_median"] = df[home_columns].median(axis=1)

#print(df)
#df = df[df[odds_columns].isnull().sum(axis=1) <= 10]

#print(df)
for away_col, home_col in zip(away_columns, home_columns):
    df[away_col] = df[away_col].fillna(df["away_median"])
    df[home_col] = df[home_col].fillna(df["home_median"])


df.to_sql("median_book_odds2", conn, if_exists="replace", index=False)

# Close the database connection

df = df[df["home_is_winner"] != "Unknown"]
df = df[df["away_is_winner"] != "Unknown"]
df = df[df["home_score"] != "Unknown"]

print(df)

#df.to_csv('median_odds2.csv')
conn.close()
'''