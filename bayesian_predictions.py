import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def run_predictions(cols, model_uri):
    for i in range(1,100):
        # Load test data
        data = pd.read_csv(f'{model_uri}/sigmoid_test_data_{i}.csv')
        data = data[data['home_is_winner'] != 'Unknown']
        
        print(data.columns)
        print(cols)
        X= data[cols]

        # Load the model
        with open(f'bayesian_tests/experiment_21/model/no_calibration/trained_model_{i}.pkl', 'rb') as file:
            nc_model = pickle.load(file)
        with open(f'bayesian_tests/experiment_21/model/isotonic/trained_model_{i}.pkl', 'rb') as file:
            isotonic_model = pickle.load(file)
        with open(f'bayesian_tests/experiment_21/model/sigmoid/trained_model_{i}.pkl', 'rb') as file:
            sigmoid_model = pickle.load(file)

        # Predict probabilities using the calibrated model
        nc_probs = nc_model.predict_proba(X)
        isotonic_probs = isotonic_model.predict_proba(X)
        sigmoid_probs = sigmoid_model.predict_proba(X)

        # Initialize lists for each model
        no_calibration_preds =[]
        isotonic_preds =[]
        sigmoid_preds =[]

        # Iterate over games
        for idx, (_, row) in enumerate(data.iterrows()):
            game_id = row['id']
            game_date = row['game_date']
            home_odds = row['bovada_home']
            away_odds = row['bovada_away']

            # Implied probabilities from the odds
            home_implied_prob = abs(home_odds) / (abs(home_odds) + 100) if home_odds < 0 else 100 / (home_odds + 100)
            away_implied_prob = abs(away_odds) / (abs(away_odds) + 100) if away_odds < 0 else 100 / (away_odds + 100)

            # Calculate model probabilities
            nc_home_win_prob = (nc_probs[idx, 1])
            nc_away_win_prob = (nc_probs[idx, 0])
            isotonic_home_win_prob = (isotonic_probs[idx, 1])
            isotonic_away_win_prob = (isotonic_probs[idx, 0])
            sigmoid_home_win_prob = (sigmoid_probs[idx, 1])
            sigmoid_away_win_prob =-(sigmoid_probs[idx, 0])

            # Append this game's predictions to the lists
            no_calibration_preds.append([game_id, game_date, row['home_team'], row['away_team'], row['home_is_winner'], 
                                         home_odds, away_odds, home_implied_prob, away_implied_prob,
                                         nc_home_win_prob, nc_away_win_prob])
            isotonic_preds.append([game_id, game_date, row['home_team'], row['away_team'], row['home_is_winner'], 
                                   home_odds, away_odds, home_implied_prob, away_implied_prob,
                                   isotonic_home_win_prob, isotonic_away_win_prob])
            sigmoid_preds.append([game_id, game_date, row['home_team'], row['away_team'], row['home_is_winner'], 
                                  home_odds, away_odds, home_implied_prob, away_implied_prob,
                                  sigmoid_home_win_prob, sigmoid_away_win_prob])

        # Create DataFrame from the list and save it as a CSV
        df_list = [no_calibration_preds, isotonic_preds, sigmoid_preds]
        df_names = ['no_calibration', 'isotonic', 'sigmoid']
        for preds, name in zip(df_list, df_names):
            df = pd.DataFrame(preds, columns=[
                    'id', 'game_date', 'home_team', 'away_team', 'home_is_winner', 
                    'home_odds', 'away_odds', 'home_implied_prob', 'away_implied_prob',
                    'model_home_win_prob', 'model_away_win_prob'])
            df.to_csv(f'bayesian_tests/experiment_21/predictions/{name}/test_predictions_{i}.csv', index=False)