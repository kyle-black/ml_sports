import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Load and preprocess your dataset


# Define X and y variables


data = pd.read_csv('resultodds_new.csv')

# Define X and y variables
data = data[data['home_is_winner'] != 'Unknown']
test_data = data[data['season']== 2023]
data = data[data['season'] != 2023]

data = data[data['home_is_winner'] != 'Unknown']
X = data.drop(['game_id','home_team','away_team','commence_time','id','game_date','season','home_score','away_score','away_is_winner','game_type','home_is_winner', 'betfair_away', 'betfair_home', 'caesars_away', 'caesars_home'], axis=1)
#print(X.columns)
print(X.columns)

y = data['home_is_winner'].astype(int)
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)
X_scaled =X
#print(X.columns)

# Apply PCA to reduce dimensionality
#pca = PCA(n_components=0.95, svd_solver='full')
#X_reduced = pca.fit_transform(X_scaled)

#print(X_reduced.columns)


#with open('pca.pkl', 'wb') as f:
#    pickle.dump((pca), f)

# Split the reduced data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the parameter grids for each classifier
#param_grid_rf = {'n_estimators': [10, 50, 100, 150], 'max_depth': [None, 10, 20, 30]}
#param_grid_ada = {'n_estimators': [50, 100, 150], 'learning_rate': [0.5, 1.0, 1.5]}
#param_grid_gb = {'n_estimators': [50, 100, 150], 'learning_rate': [0.5, 1.0, 1.5], 'max_depth': [1, 2, 3]}
#param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
#param_grid_knn = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
param_grid_xgb = {
    'max_depth': [3, 4, 5],
    'min_child_weight': [1, 5, 10],
    'gamma': [0.5, 1, 1.5, 2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'lambda': [0, 1, 5, 10]
}

# Set up GridSearchCV objects for each classifier
#grid_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
#grid_ada = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=param_grid_ada, cv=5, scoring='accuracy', n_jobs=-1)
#grid_gb = GridSearchCV(estimator=GradientBoostingClassifier(random_state=0), param_grid=param_grid_gb, cv=5, scoring='accuracy', n_jobs=-1)
#grid_svm = GridSearchCV(estimator=SVC(probability=True), param_grid=param_grid_svm, cv=5, scoring='accuracy', n_jobs=-1)
#grid_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid_knn, cv=5, scoring='accuracy', n_jobs=-1)
grid_xgb = GridSearchCV(estimator=XGBClassifier(use_label_encoder=False, eval_metric='logloss'), param_grid=param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1)

# Perform grid search on training
# Perform grid search on training data for each classifier
#grid_rf.fit(X_train, y_train)
#grid_ada.fit(X_train, y_train)
#grid_gb.fit(X_train, y_train)
#grid_svm.fit(X_train, y_train)
#grid_knn.fit(X_train, y_train)
grid_xgb.fit(X_train, y_train)

# Get the best parameters for each classifier
#best_params_rf = grid_rf.best_params_
#best_params_ada = grid_ada.best_params_
#best_params_gb = grid_gb.best_params_
#best_params_svm = grid_svm.best_params_
#best_params_knn = grid_knn.best_params_
best_params_xgb = grid_xgb.best_params_

#print("Best parameters for RandomForest:", best_params_rf)
#print("Best parameters for AdaBoost:", best_params_ada)
#print("Best parameters for GradientBoosting:", best_params_gb)
#print("Best parameters for SVM:", best_params_svm)
#print("Best parameters for kNN:", best_params_knn)
print("Best parameters for XGBoost:", best_params_xgb)

# Create individual classifiers with the best parameters
#clf1 = RandomForestClassifier(**best_params_rf, random_state=42)
#clf2 = AdaBoostClassifier(**best_params_ada)
#clf3 = GradientBoostingClassifier(**best_params_gb, random_state=0)
#clf4 = SVC(**best_params_svm, probability=True)
#clf5 = KNeighborsClassifier(**best_params_knn)
clf6 = XGBClassifier(**best_params_xgb, use_label_encoder=False, eval_metric='logloss')

# Combine individual classifiers using VotingClassifier
#eclf = VotingClassifier(estimators=[ ('clf4', clf4), ('clf5', clf5)], voting='soft')

# Train and evaluate the model
clf6.fit(X_train, y_train)
calibrator = CalibratedClassifierCV(clf6, method='sigmoid', cv=5)
calibrator.fit(X_train, y_train)

# Now you can predict probabilities using the calibrator instead of the original classifier
calibrated_probs = calibrator.predict_proba(X_test)
print("Probs:", calibrated_probs)

y_pred_proba = calibrated_probs
#y_pred = clf6.predict(X_test)
#y_pred_proba = clf6.predict_proba(X_test)
#print('predictions:', y_pred)
#print("Probs:", y_pred_proba)

#print("Accuracy:", accuracy_score(y_test, y_pred))
#print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

with open('trained_model4.pkl', 'wb') as file:
    pickle.dump((clf6), file)

test_data = data.loc[y_test.index]

# Combine the predictions and probabilities with the original test dataset
results_df = pd.DataFrame({'home_team': test_data['home_team'],
                           'away_team': test_data['away_team'],
                           'home_is_winner': y_test,
                           'away_win_prob': y_pred_proba[:, 0],
                           'home_win_prob': y_pred_proba[:, 1]})

# Print the results
print(results_df)
brier_score = brier_score_loss(y_test, y_pred_proba[:, 1])
print('Brier Score:', brier_score)


logloss = log_loss(y_test, y_pred_proba[:, 1])
print('Log Loss:', logloss)



prob_pos = calibrated_probs[:, 1]

# Get the true outcomes
true_outcomes = y_test

# Generate the calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(true_outcomes, prob_pos, n_bins=10)

# Plot the calibration curve
plt.plot(mean_predicted_value, fraction_of_positives, "s-")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.title('Calibration curve')
plt.show()
try:
    plt.savefig('cal_fig.png')
except Exception as e:
    print(e)




results_df.to_csv('results4.csv')
test_data.to_csv('2023_test_data.csv')
