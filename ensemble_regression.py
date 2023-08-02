from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import pandas as pd
from sklearn.dummy import DummyRegressor
import numpy as np

# Load and preprocess your dataset
data = pd.read_csv('resultodds_new7.csv')

data = data[data['season'] ==2023]
data = data[data['home_score']!= 'Unknown']
data['home_score'] = pd.to_numeric(data['home_score'], errors='coerce')

#scaler = StandardScaler()
#X = data[['home_R','away_R','home_RA','away_RA','home_W-L%', 'away_W-L%']]
X =data[['pinnacle_home', 'pinnacle_away','lowvig_home', 'lowvig_away',]]

#X_scaled = scaler.fit_transform(X)
X_scaled =X
print(X_scaled)


y = data['home_score']

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=80)

# Grid search parameters
param_grid_rf = {'n_estimators': [120,130,140,180,190,210,200,220], 'max_depth': [1,2,3,4,5,7,8,9]}
param_grid_ada = {'n_estimators': [5,10,15,20,30,35,40,45,50], 'learning_rate': [0.5,.10,.15,.18,.22,.25]}
param_grid_gb = {'n_estimators': [5,10,15,20,25,30,35,40,45,50], 'learning_rate': [.01,.05,.1,.15,.18,.22,.25], 'max_depth': [1,2,3,4,5,6,8,9]}
param_grid_svc = {'C': [0.8,0.9,1.1,1.2, 1.3,1.4,1.5,1.6], 'gamma': ['scale', 'auto']}
param_grid_knn = {'n_neighbors': [6,7,8,9,10,11,12]}
param_grid_xgb = {'n_estimators': [5,10,15,20,30,35,40,45,50], 'learning_rate': [.01,.05,.1,.15,.18,.22,.25]}

# Grid search classifiers
grid_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid_rf, cv=5, scoring='r2', n_jobs=-1)
grid_ada = GridSearchCV(estimator=AdaBoostRegressor(), param_grid=param_grid_ada, cv=5, scoring='r2', n_jobs=-1)
grid_gb = GridSearchCV(estimator=GradientBoostingRegressor(random_state=0), param_grid=param_grid_gb, cv=5, scoring='r2', n_jobs=-1)
grid_svc = GridSearchCV(estimator=SVR(), param_grid=param_grid_svc, cv=5, scoring='r2', n_jobs=-1)
grid_knn = GridSearchCV(estimator=KNeighborsRegressor(), param_grid=param_grid_knn, cv=5, scoring='r2', n_jobs=-1)
grid_xgb = GridSearchCV(estimator=XGBRegressor(eval_metric="rmse"), param_grid=param_grid_xgb, cv=5, scoring='r2', n_jobs=-1)

# Fit grid search classifiers
grid_rf.fit(X_train, y_train)
grid_ada.fit(X_train, y_train)
grid_gb.fit(X_train, y_train)
grid_svc.fit(X_train, y_train)
grid_knn.fit(X_train, y_train)
grid_xgb.fit(X_train, y_train)

# Check the best parameters
print("Best parameters for RandomForest: ", grid_rf.best_params_)
print("Best parameters for AdaBoost: ", grid_ada.best_params_)
print("Best parameters for GradientBoosting: ", grid_gb.best_params_)
print("Best parameters for SVC: ", grid_svc.best_params_)
print("Best parameters for KNN: ", grid_knn.best_params_)
print("Best parameters for XGBRegressor: ", grid_xgb.best_params_)

# Apply the best parameters to the classifiers
clf1 = RandomForestRegressor(**grid_rf.best_params_, random_state=42)
clf2 = AdaBoostRegressor(**grid_ada.best_params_)
clf3 = GradientBoostingRegressor(**grid_gb.best_params_)
clf4 = SVR(**grid_svc.best_params_)
clf5 = KNeighborsRegressor(**grid_knn.best_params_)
clf6 = XGBRegressor(**grid_xgb.best_params_, eval_metric="rmse")

# Establish a baseline with the DummyRegressor
dummy = DummyRegressor(strategy='mean')

# Dictionary of classifiers
classifiers = {'Dummy': dummy, 'RF': clf1, 'AdaBoost': clf2, 'GradientBoost': clf3, 'SVC': clf4, 'KNN': clf5, 'XGB': clf6}

# Train classifiers and evaluate on the test set
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f'{name}:')
    print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
    print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    print('R2 Score:', r2_score(y_test, y_pred))
    print(y_pred)

# Save the models
for name, clf in classifiers.items():
    pickle.dump(clf, open(name + '.pkl', 'wb'))
