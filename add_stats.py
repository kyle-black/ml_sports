import requests
from bs4 import BeautifulSoup
import pandas as pd

# Generate dates
dates = pd.date_range(start='2022-03-13', end='2022-10-05')

# Create an empty DataFrame
df_all_dates = pd.DataFrame(columns=['Date' ,'Team', 'Current','Last 3', 'Last 1', 'Home', 'Away', 'Previous'])

# Loop over dates
for date in dates:
    # Convert date to the string format for URL
    date_str = date.strftime('%Y-%m-%d')

    url = f"https://www.teamrankings.com/mlb/stat/win-pct-all-games?date={date_str}"
    response = requests.get(url)

    soup = BeautifulSoup(response.content, 'html.parser')

    try:
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
            # Create a new DataFrame from the data and append it to the main DataFrame
            df_all_dates = pd.concat([df_all_dates, pd.DataFrame([data])], ignore_index=True)

    except AttributeError:
        print('Table not found on {}'.format(date_str))

print(df_all_dates)
df_all_dates.to_csv('date_stats/win_stats/2022_stats_winpct.csv')