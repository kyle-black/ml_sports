from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, brier_score_loss, log_loss
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import pickle

# Load and preprocess your dataset
data = pd.read_csv('filled_csv.csv')

# Define X and y variables
data = data[data['home_is_winner'] != 'Unknown']
X = data.drop(['game_id','home_team','away_team','commence_time','id','game_date','season','home_score','away_score','away_is_winner','game_type','home_is_winner', 'betfair_away', 'betfair_home', 'caesars_away', 'caesars_home'], axis=1)
y = data['home_is_winner'].astype(int)


#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=0.95, svd_solver='full')
X_reduced = pca.fit_transform(X)
#X_reduced =X
#with open('pca.pkl', 'wb') as f:
#    pickle.dump((pca), f)

# Split the reduced data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Define the parameter grids for each classifier
param_grid_rf = {'n_estimators': [10, 50, 100, 150], 'max_depth': [None, 10, 20, 30]}
param_grid_ada = {'n_estimators': [50, 100, 150], 'learning_rate': [0.5, 1.0, 1.5]}
param_grid_gb = {'n_estimators': [50, 100, 150], 'learning_rate': [0.5, 1.0, 1.5], 'max_depth': [1, 2, 3]}
param_grid_ridge = {'alpha': [0.1, 1, 10, 100]}

# Set up GridSearchCV objects for each classifier
grid_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_rf, cv=5, scoring='neg_log_loss', n_jobs=-1)
grid_ada = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=param_grid_ada, cv=5, scoring='neg_log_loss', n_jobs=-1)
grid_gb = GridSearchCV(estimator=GradientBoostingClassifier(random_state=0), param_grid=param_grid_gb, cv=5, scoring='neg_log_loss', n_jobs=-1)
grid_ridge = GridSearchCV(estimator=RidgeClassifier(), param_grid=param_grid_ridge, cv=5, scoring='neg_log_loss', n_jobs=-1)

# Perform grid search on training data for each classifier
grid_rf.fit(X_train, y_train)
grid_ada.fit(X_train, y_train)
grid_gb.fit(X_train, y_train)
grid_ridge.fit(X_train, y_train)

# Get the best parameters for each classifier
best_params_rf = grid_rf.best_params_
best_params_ada = grid_ada.best_params_
best_params_gb = grid_gb.best_params_
best_params_ridge = grid_ridge.best_params_

print("Best parameters for RandomForest:", best_params_rf)
print("Best parameters for AdaBoost:", best_params_ada)
print("Best parameters for GradientBoosting:", best_params_gb)
print("Best parameters for RidgeClassifier:", best_params_ridge)

# Create individual classifiers with the best parameters
clf1 = RandomForestClassifier(**best_params_rf, random_state=42)
clf2 = AdaBoostClassifier(**best_params_ada)
clf3 = GradientBoostingClassifier(**best_params_gb, random_state=0)

# Combine individual classifiers using VotingClassifier
eclf = VotingClassifier(estimators=[('rf', clf1), ('clf2', clf2), ('clf3', clf3)], voting='soft')

# Train and evaluate the model
eclf.fit(X_train, y_train)
y_pred = eclf.predict(X_test)
y_pred_proba = eclf.predict_proba(X_test)

print('predictions:', y_pred)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Brier Score:", brier_score_loss(y_test, y_pred_proba[:, 1]))
print("Log Loss:", log_loss(y_test, y_pred_proba))

with open('trained_model.pkl', 'wb') as file:
    pickle.dump((eclf), file)