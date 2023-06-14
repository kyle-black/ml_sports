import pandas as pd
import sqlite3


conn = sqlite3.connect('mlb_games.db')


sql_query =sql_query = '''
SELECT * FROM book_odds9 JOIN past_games_update2 as pg 
WHERE (book_odds9.home_team = pg.home_team) 
AND (date(book_odds9.commence_time) = date(pg.commence_time));
'''



df = pd.read_sql_query(sql_query, conn)
# Load CSV data into a DataFrame



# Drop rows where all the data is null
df = df.dropna(how='all', subset=df.columns[5:])

# Compute the median for home and away odds
medians_home = df.filter(like='_home').median()
medians_away = df.filter(like='_away').median()

# Fill null values with the corresponding median
for col in df.columns:
    if col.endswith('_home'):
        df[col].fillna(medians_home[col], inplace=True)
    elif col.endswith('_away'):
        df[col].fillna(medians_away[col], inplace=True)

print(df)

df.to_csv('resultodds_new.csv')
conn.close()