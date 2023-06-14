import sqlite3
import pandas as pd
import ast
# Connect to the SQLite database
conn = sqlite3.connect('mlb_games.db')

# Define the SQL query
sql_query = '''
SELECT book_odds5.*, past_games.* FROM book_odds5 JOIN past_games ON (book_odds5.home_team = past_games.home_team) AND (book_odds5.commence_time = past_games.commence_time);
'''

# Execute the query and fetch the results as a DataFrame
df = pd.read_sql_query(sql_query, conn)

#print(df['lowvig'][0])


def parse_odds(odds_string):
    if pd.isna(odds_string):
        return None
    try:
        odds_string = odds_string.replace('[', '').replace(']', '')
        return [int(x) for x in odds_string.split(',')]
    except (ValueError, SyntaxError):
        return None

odds_columns = ['lowvig', 'betonlineag', 'unibet', 'draftkings', 'pointsbetus', 'gtbets', 'mybookieag', 'bovada', 'fanduel', 'betfair', 'intertops', 'williamhill_us', 'betrivers', 'betmgm', 'caesars', 'sugarhouse', 'foxbet', 'barstool', 'twinspires', 'betus', 'wynnbet', 'circasports', 'superbook', 'unibet_us']
df[odds_columns] = df[odds_columns].applymap(parse_odds)

print(df['lowvig'][0])
'''
def median_odds(row, columns):
    odds_values = [x for col in columns for x in (row[col] if row[col] is not None else []) if isinstance(row[col], list)]
    return pd.Series(odds_values).median()

for col in odds_columns:
    df.loc[df[col].notna(), col] = df[df[col].notna()].apply(lambda row: [median_odds(row, odds_columns)] * 2, axis=1)

# Split odds columns into away and home columns
for col in odds_columns:
    df[f"{col}_away"] = df[col].apply(lambda x: x[0] if x is not None else None)
    df[f"{col}_home"] = df[col].apply(lambda x: x[1] if x is not None else None)

# Drop original odds columns
df = df.drop(columns=odds_columns)

away_odds_columns = [f"{col}_away" for col in odds_columns]
home_odds_columns = [f"{col}_home" for col in odds_columns]
all_odds_columns = away_odds_columns + home_odds_columns
df = df[df[all_odds_columns].isnull().sum(axis=1) <= 5]

# Replace null values with the median of the corresponding home or away team odds
for col in away_odds_columns:
    median_away = df[col].median()
    df[col] = df[col].fillna(median_away)

for col in home_odds_columns:
    median_home = df[col].median()
    df[col] = df[col].fillna(median_home)

print(df)

# Save the DataFrame to the SQLite database
df.to_sql("median_book_odds", conn, if_exists="replace", index=False)

# Close the database connection
df.to_csv('median_odds.csv')
conn.close()
'''