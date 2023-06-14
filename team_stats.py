import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Connect to the SQLite database
engine = create_engine("sqlite:///mlb_games.db")
Session = sessionmaker(bind=engine)
session = Session()

# Read the CSV file into a pandas DataFrame
csv_file = "data/baseball/mlb/team_stats/2019/batting/batting_2019.csv"  # Replace this with the path to your CSV file
df = pd.read_csv(csv_file)

# Add a new column for the season
df["season"] = 2019

# Rename columns in the DataFrame if necessary
'''
column_mapping = {
    "csv_column_name1": "team_stats_column_name1",
    "csv_column_name2": "team_stats_column_name2",

    # Add more mappings for the rest of the columns
}
df.rename(columns=column_mapping, inplace=True)
'''
# Write the DataFrame to the SQLite database
df.to_sql("team_batting", engine, if_exists="append", index=False)

# Commit and close the session
session.commit()
session.close()