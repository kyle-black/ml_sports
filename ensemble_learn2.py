import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import Counter


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def XGBoost_train(X, y, cols, df):

    X = X[cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid_xgb = {
        'max_depth': [3, 4, 5],
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'lambda': [0, 1, 5, 10]
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_xgb = GridSearchCV(estimator=XGBRegressor(objective='reg:squarederror'), 
                            param_grid=param_grid_xgb, 
                            cv=kf, 
                            scoring='neg_mean_squared_error', 
                            n_jobs=-1)

    grid_xgb.fit(X_train, y_train)

    best_params_xgb = grid_xgb.best_params_
    print("Best parameters for XGBoost:", best_params_xgb)

    clf6 = XGBRegressor(**best_params_xgb, objective='reg:squarederror')
    clf6.fit(X_train, y_train)

    predictions = clf6.predict(X_test)
    print("Predictions:", predictions)

    test_data = df.loc[y_test.index]

    results_df = pd.DataFrame({'home_team': test_data['home_team'],
                               'away_team': test_data['away_team'],
                               'home_is_winner': y_test,
                               'home_win_prob': predictions})

    print(results_df)

    mse = mean_squared_error(y_test, predictions)
    print('Mean Squared Error:', mse)

    with open('XGboost/experiment_1/model/trained_model_1.pkl', 'wb') as f:
        pickle.dump(clf6, f)

    results_df.to_csv('XGBoost/experiment_1/test_data/test_data_1.csv')

    return results_df