from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score, make_scorer, brier_score_loss
from sklearn.dummy import DummyClassifier
from sklearn.metrics import brier_score_loss, f1_score, log_loss, precision_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree




brier_scorer = make_scorer(brier_score_loss, greater_is_better=False, needs_proba=True)
# Load and preprocess your dataset
data = pd.read_csv('resultodds_new7.csv')
data = data[data['home_is_winner'] != 'Unknown']
#test_data = data[data['season']== 2023]
#data = data[data['season']!= 2023]
#test_data = data[data['season']== 2023]
#test_data.to_csv('2023_test_data.csv')
#data.to_csv('train_data_pull.csv')

# Your feature columns
'''
X = data[['lowvig_home', 'lowvig_away', 'betonlineag_home', 'betonlineag_away',
          'draftkings_home', 'draftkings_away',
          'pointsbetus_home', 'pointsbetus_away',
          'mybookieag_home', 'mybookieag_away', 'bovada_home', 'bovada_away',
          'fanduel_home', 'fanduel_away',
          'williamhill_us_home', 'williamhill_us_away', 'betrivers_home',
          'betrivers_away', 'betmgm_home', 'betmgm_away', 'sugarhouse_home',
          'sugarhouse_away', 'foxbet_home', 'foxbet_away', 'barstool_home',
          'barstool_away', 'twinspires_home', 'twinspires_away', 'betus_home',
          'betus_away', 'wynnbet_home', 'wynnbet_away', 'circasports_home',
          'circasports_away', 'superbook_home', 'superbook_away', 'pinnacle_home','pinnacle_away']]
'''
#X = data[['pinnacle_home','pinnacle_away','lowvig_home', 'lowvig_away', 'betonlineag_home',
 #      'betonlineag_away', 'draftkings_home', 'draftkings_away','pinnacle_vig', 'lowvig_vig', 'betonlineag_vig','draftkings_vig']]
#X = data[['betonlineag_home','betonlineag_away']]

X = data[['pinnacle_home', 'pinnacle_away','lowvig_home', 'lowvig_away']]
#X=data[['home_R','away_R','home_RA','away_RA' , 'home_W-L%', 'away_W-L%']]
#X =data[['lowvig_home_prob', 'lowvig_away_prob','lowvig_vig']]

# Standardize your features
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

X_scaled =X
y = data['home_is_winner'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=80)

# Grid search parameters
param_grid_rf = {'n_estimators': [120,130,140,180,190,210,200,220], 'max_depth': [1,2,3,4,5,7,8,9]}
param_grid_ada = {'n_estimators': [5,10,15,20,30,35,40,45,50], 'learning_rate': [0.5,.10,.15,.18,.22,.25]}
param_grid_gb = {'n_estimators': [5,10,15,20,25,30,35,40,45,50], 'learning_rate': [.01,.05,.1,.15,.18,.22,.25], 'max_depth': [1,2,3,4,5,6,8,9]}
param_grid_svc = {'C': [0.8,0.9,1.1,1.2, 1.3,1.4,1.5,1.6], 'gamma': ['scale', 'auto']}
param_grid_knn = {'n_neighbors': [6,7,8,9,10,11,12]}
param_grid_xgb = {'n_estimators': [5,10,15,20,30,35,40,45,50], 'learning_rate': [.01,.05,.1,.15,.18,.22,.25]}

# Grid search classifiers
grid_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_rf, cv=5, scoring=brier_scorer, n_jobs=-1)
grid_ada = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=param_grid_ada, cv=5, scoring=brier_scorer, n_jobs=-1)
grid_gb = GridSearchCV(estimator=GradientBoostingClassifier(random_state=0), param_grid=param_grid_gb, cv=5, scoring=brier_scorer, n_jobs=-1)
grid_svc = GridSearchCV(estimator=SVC(probability=True, random_state=0), param_grid=param_grid_svc, cv=5, scoring=brier_scorer, n_jobs=-1)
grid_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid_knn, cv=5, scoring=brier_scorer, n_jobs=-1)
grid_xgb = GridSearchCV(estimator=XGBClassifier(use_label_encoder=False, eval_metric="logloss"), param_grid=param_grid_xgb, cv=5, scoring=brier_scorer, n_jobs=-1)

# Fit grid search classifiers
grid_rf.fit(X_train, y_train)
grid_ada.fit(X_train, y_train)
grid_gb.fit(X_train, y_train)
grid_svc.fit(X_train, y_train)
grid_knn.fit(X_train, y_train)
grid_xgb.fit(X_train, y_train)

print("Best parameters for RandomForest: ", grid_rf.best_params_)
print("Best parameters for AdaBoost: ", grid_ada.best_params_)
print("Best parameters for GradientBoosting: ", grid_gb.best_params_)
print("Best parameters for SVC: ", grid_svc.best_params_)
print("Best parameters for KNeighbors: ", grid_knn.best_params_)
print("Best parameters for XGB: ", grid_xgb.best_params_)

# Use best parameters for classifiers
clf1 = RandomForestClassifier(**grid_rf.best_params_, random_state=42)
clf2 = AdaBoostClassifier(**grid_ada.best_params_)
clf3 = GradientBoostingClassifier(**grid_gb.best_params_, random_state=0)
clf4 = SVC(**grid_svc.best_params_, probability=True, random_state=0)
clf5 = KNeighborsClassifier(**grid_knn.best_params_)
clf6 = XGBClassifier(**grid_xgb.best_params_, use_label_encoder=False, eval_metric="logloss")


#clf6.fit(X_train,y_train)

#importances_xgb = clf6.feature_importances_
#feature_importances_xgb = pd.DataFrame({'feature': X.columns, 'importance_xgb': importances_xgb})
#feature_importances_xgb = feature_importances_xgb.sort_values('importance_xgb', ascending=False)
#print(feature_importances_xgb)
# Voting Classifier
'''
eclf = VotingClassifier(estimators=[('rf', clf1), ('ada', clf2), ('gb', clf3), ('svc', clf4), ('knn', clf5), ('xgb', clf6)], voting='soft')

#eclf = VotingClassifier(estimators=[('xgb', clf6)], voting ='soft')
eclf.fit(X_train, y_train)

y_pred = eclf.predict(X_test)
y_probs = eclf.predict_proba(X_test)
#y_pred = clf6.predict(X_test)
#y_probs = clf6.predict_proba(X_test)
print("Precision:", precision_score(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Add F1 score here
print('F1 Score:', f1_score(y_test, y_pred))

brier_score = brier_score_loss(y_test, y_probs[:, 1])
print('Brier Score:', brier_score)

# Compute Log loss
log_loss_score = log_loss(y_test, y_probs)
print('Log Loss:', log_loss_score)

# Save the trained model and the scaler
with open('trained_model_10.pkl', 'wb') as file:
    pickle.dump((eclf), file)



#dummy_clf = DummyClassifier(strategy='uniform')
dummy_clf = DummyClassifier(strategy='constant', constant=0)  # This will generate predictions uniformly at random.
dummy_clf.fit(X_train, y_train)


dummy_probs = dummy_clf.predict_proba(X_test)

# Compute Brier score for the dummy classifier
dummy_brier_score = brier_score_loss(y_test, dummy_probs[:, 1])

print('Brier Score for baseline classifier:', dummy_brier_score)

constant_proba = np.full((len(X_test), 2), 0.5)

# Compute Brier score for the constant probabilities
constant_brier_score = brier_score_loss(y_test, constant_proba[:, 1])
print('Brier Score for constant probabilities:', constant_brier_score)



test_indicies = X_test.index

test_data_df = data.loc[test_indicies]

'''
#test_data_df.to_csv('test_data_df2.csv')
#print(test_data_df)

dummy_clf = DummyClassifier(strategy='uniform')
dummy_clf = DummyClassifier(strategy='constant', constant=0)  # This will generate predictions uniformly at random.
constant_proba = np.full((len(X_test), 2), 0.5)
dummy_clf.fit(X_train, y_train)

classifiers = {'Dummy': dummy_clf, 'RF': clf1, 'AdaBoost': clf2, 'GradientBoost': clf3, 'SVC': clf4, 'KNN': clf5, 'XGB': clf6}


for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_probs = clf.predict_proba(X_test)
    print(f'{name}:')
    #print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
    #print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    #print('R2 Score:', r2_score(y_test, y_pred))
    #print(y_pred)
    print(y_probs)
    print('brier_score:', brier_score_loss(y_test, y_probs[:, 1]))
    print(clf.classes_)
feature_importance = clf1.feature_importances_
features = X_train.columns
estimator = clf1.estimators_[5] 

plt.figure(figsize=(20,10))  
plot_tree(estimator, filled=True)
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(features, feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()