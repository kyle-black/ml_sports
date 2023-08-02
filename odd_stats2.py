import pandas as pd

import sqlite3 

import requests
from bs4 import BeautifulSoup
import pandas as pd



db_connect = 'mlb_games.db'
conn =sqlite3.connect(db_connect)


#win_pct =pd.read_csv('date_stats/winpct.csv')

#win_pct.to_sql('winpct',conn)




# Generate dates
dates = pd.date_range(start='2007-03-30', end='2007-10-05')

# Create an empty DataFrame
df_all_dates = pd.DataFrame(columns=['Date' ,'Team', '2007','Last 3', 'Last 1', 'Home', 'Away', '2006'])

# Loop over dates
for date in dates:
    # Convert date to the string format for URL
    date_str = date.strftime('%Y-%m-%d')
    
    url = f"https://www.teamrankings.com/mlb/stat/win-pct-all-games?date={date_str}"
    response = requests.get(url)

    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table', class_='tr-table datatable scrollable')

    rows = table.tbody.find_all('tr')

    for row in rows:
        columns = row.find_all('td')

        data = {
            'Date': date_str,
            'Team': columns[1].text,
            'Current': columns[2].text,
            'Last 3': columns[3].text,
            'Last 1': columns[4].text,
            'Home': columns[5].text,
            'Away': columns[6].text,
            'Previous': columns[7].text
        }
        print(data)
        # Create a new DataFrame from the data and append it to the main DataFrame
        df_all_dates = pd.concat([df_all_dates, pd.DataFrame([data])], ignore_index=True)

print(df_all_dates)
df_all_dates.to_csv('date_stats/2007_stats_winpct.csv')

df_all_dates.to_sql('winpct',conn, if_exists='append')


