# Import necessary libraries
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import data_parse

# Connect to the database
engine = create_engine('sqlite:///mlb_games.db')

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Query the data from the table
games = session.query(data_parse.Game)

# Process the results
for game in games:
    print(f" Home Team: {game.home_team}, Away Team: {game.away_team}, Season{game.season}")

# Close the session
session.close()
