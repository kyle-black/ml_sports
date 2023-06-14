import pandas as pd
import pickle
from sklearn.calibration import CalibratedClassifierCV

# Load test data
data = pd.read_csv('2023_test_data.csv')

# Define X and y variables
data = data[data['home_is_winner'] != 'Unknown']
test_data = data[data['season']== 2023]
data = data[data['season'] != 2023]

# Define the X_test
X_test = test_data.drop(['game_id', 'home_team', 'away_team', 'commence_time', 'home_is_winner', 
                         'away_is_winner', 'id', 'game_date', 'season', 'home_score', 'away_score', 
                         'game_type'], axis=1)

# Load the model
with open('trained_model4.pkl', 'rb') as file:
    model = pickle.load(file)

# Calibrate the classifier
calibrator = CalibratedClassifierCV(model, method='sigmoid', cv=5)
calibrator.fit(X_test, test_data['home_is_winner'].astype(int))

# Predict probabilities using the calibrator
calibrated_probs = calibrator.predict_proba(X_test)

# Initialize a new dataframe to hold game predictions and odds
predictions_and_odds = []

# Iterate over games
for idx, (_, row) in enumerate(test_data.iterrows()):
    game_id = row['game_id']
    
    # Get odds from the row
    home_odds = row['bovada_home']
    away_odds = row['bovada_away']

    # Implied probabilities from the odds
    #home_implied_prob = 100 / (-home_odds) if home_odds < 0 else 100 / (home_odds + 100)
    #away_implied_prob = 100 / (-away_odds) if away_odds < 0 else 100 / (away_odds + 100)

    home_implied_prob = abs(home_odds) / (abs(home_odds) + 100) if home_odds < 0 else 100 / (home_odds + 100)
    away_implied_prob = abs(away_odds) / (abs(away_odds) + 100) if away_odds < 0 else 100 / (away_odds + 100)
    
    # Your predicted probabilities
    home_win_prob = calibrated_probs[idx, 1]
    away_win_prob = calibrated_probs[idx, 0]

    # Append this game's predictions and odds to the dataframe
    predictions_and_odds.append([game_id, row['home_team'], row['away_team'], row['home_is_winner'], 
                                home_odds, away_odds, home_implied_prob, away_implied_prob,
                                home_win_prob, away_win_prob])

# Create DataFrame from the list
results_df = pd.DataFrame(predictions_and_odds, columns=[
    'game_id', 'home_team', 'away_team', 'home_is_winner', 
    'home_odds', 'away_odds', 'home_implied_prob', 'away_implied_prob',
    'home_win_prob', 'away_win_prob'])

# Save the dataframe to a csv file
results_df.to_csv('game_predictions_and_odds2.csv', index=False)