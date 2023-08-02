# Import necessary libraries
import json
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Sequence
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import requests
from datetime import datetime

# Define the table structure using SQLAlchemy ORM
Base = declarative_base()

class Game(Base):
    __tablename__ = 'past_games_update_4'

    id = Column(Integer, Sequence('game_id_seq'), primary_key=True)
    game_date = Column(DateTime)
    season = Column(String)
    home_team = Column(String)
    away_team = Column(String)
    home_score = Column(Integer)
    away_score = Column(Integer)
    home_is_winner = Column(String)
    away_is_winner = Column(String)
    game_type = Column(String)
    commence_time = Column(String)

# Connect to the database
engine = create_engine('sqlite:///mlb_games.db')
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Delete all rows from the table
#session.query(Game).delete()

# Commit the changes
#session.commit()

# Insert the data into the table
start_date = '2023-01-01'
end_date = '2023-07-05'
data = requests.get(f'https://statsapi.mlb.com/api/v1/schedule?startDate={start_date}&endDate={end_date}&sportId=1')
json_data = data.json()
dates = json_data['dates']

for date in dates:
    game_date = datetime.strptime(date['date'], "%Y-%m-%d")
    games = date['games']
    for game in games:
        season = game['season']
        game_type = game['gameType']
        home_team = game['teams']['home']['team'].get('name', 'Unknown')
        away_team = game['teams']['away']['team'].get('name', 'Unknown')
        home_score = game['teams']['home'].get('score', 'Unknown')
        away_score = game['teams']['away'].get('score', 'Unknown')
        home_is_winner = game['teams']['home'].get('isWinner','Unknown')
        away_is_winner = game['teams']['away'].get('isWinner','Unknown')
        commence_time = game['officialDate']

        new_game = Game(
            game_date=game_date,
            season=season,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            home_is_winner=home_is_winner,
            away_is_winner=away_is_winner,
            game_type=game_type,
            commence_time= commence_time
        )

        session.add(new_game)

# Commit the changes
session.commit()

# Close the session

session.close()