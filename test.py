import sqlite3 
import pandas as pd


conn = sqlite3.connect('mlb_games.db')


sql_query = '''
SELECT * FROM median_book_odds2 WHERE home_is_winner !='Unknown';
'''

df = pd.read_sql_query(sql_query, conn)


df.to_csv('median_stats.csv')
print(df)