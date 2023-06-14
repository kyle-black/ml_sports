import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt

# Load and preprocess your dataset
#data = pd.read_csv('filled_csv.csv')
data = pd.read_csv('resultodds_new.csv')

# Define X and y variables
data = data[data['home_is_winner'] != 'Unknown']
test_data = data[data['season']== 2023]
data = data[data['season'] != 2023]

X = data.drop(['game_id','home_team','away_team','commence_time','id','game_date','season','home_score','away_score','away_is_winner','game_type','home_is_winner', 'betfair_away', 'betfair_home', 'caesars_away', 'caesars_home'], axis=1)
y = data['home_is_winner'].astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for XGBoost
'''
param_grid_xgb = {
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'lambda': [0, 1, 5, 10]
}
'''
param_grid_xgb = {
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'lambda': [0, 1, 5, 10],
    'learning_rate': [0.01, 0.02, 0.1],
    'n_estimators': [100, 500, 1000],
    'scale_pos_weight': [0.5, 1, 2],
    'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05],
    'max_delta_step': [0, 1, 2, 5]
}

# Set up GridSearchCV object for XGBoost
grid_xgb = GridSearchCV(estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid=param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1)

# Perform grid search on training data for XGBoost
grid_xgb.fit(X_train, y_train)

# Get the best parameters for XGBoost
best_params_xgb = grid_xgb.best_params_
print("Best parameters for XGBoost:", best_params_xgb)

# Create XGBoost classifier with the best parameters
clf = XGBClassifier(**best_params_xgb, use_label_encoder=False, eval_metric='logloss')

# Fit the classifier to the training data
clf.fit(X_train, y_train)

# Calibrate the classifier
calibrator = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
calibrator.fit(X_train, y_train)

# Predict probabilities using the calibrator
calibrated_probs = calibrator.predict_proba(X_test)
print("Calibrated Probs:", calibrated_probs)

# Compute Brier score and log loss based on the calibrated probabilities
brier_score = brier_score_loss(y_test, calibrated_probs[:, 1])
print('Brier Score:', brier_score)
logloss = log_loss(y_test, calibrated_probs[:, 1])
print('Log Loss:', logloss)

# Save the trained model
with open('trained_model3.pkl', 'wb') as file:
    pickle.dump(clf, file)

# Create a DataFrame with the test data and predicted probabilities
test_data = data.loc[y_test.index]
results_df = pd.DataFrame({
    'home_team': test_data['home_team'],
    'away_team': test_data['away_team'],
    'home_is_winner': y_test,
    'away_win_prob': calibrated_probs[:, 0],
    'home_win_prob': calibrated_probs[:, 1]
})

# Print and save the results
print(results_df)
results_df.to_csv('results5.csv')
test_data.to_csv('2023_test_data.csv')