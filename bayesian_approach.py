from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
from statistics import mean, median
import shap


#y = data['home_is_winner'].astype(int)


def bayesian_test(test_columns, X, y,total_df):
    iso_mean = []
    for i in range(1,100):
        #scaler =StandardScaler()
        #X_scaled = scaler.fit_transform(X)
        X_scaled =X
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=i)

        X_test_ = X_test
    

        # With no calibration
        clf = GaussianNB()
        clf.fit(X_train, y_train)  # GaussianNB itself does not support sample-weights
        prob_pos_clf = clf.predict_proba(X_test)[:, 1]

        # With isotonic calibration
        clf_isotonic = CalibratedClassifierCV(clf, cv=2, method="isotonic")
        clf_isotonic.fit(X_train, y_train)
        
        
        prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]





        # With sigmoid calibrationba
        clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method="sigmoid")
        clf_sigmoid.fit(X_train, y_train)
        prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]

        print("Brier score losses: (the smaller the better)")

        clf_score = brier_score_loss(y_test, prob_pos_clf)
        print("No calibration: %1.3f" % clf_score)


        clf_isotonic_score = brier_score_loss(y_test, prob_pos_isotonic)
        print("With isotonic calibration: %1.3f" % clf_isotonic_score)
        iso_mean.append(clf_isotonic_score)
        clf_sigmoid_score = brier_score_loss(y_test, prob_pos_sigmoid)
        print("With sigmoid calibration: %1.3f" % clf_sigmoid_score)


        #control_preds = (X_test['pinnacle_home_vf'] > 0.5).astype(int)

        # Calculate Brier score loss for the control model
        #control_score = brier_score_loss(y_test, control_preds)
        #print("Control: %1.3f" % control_score)
        with open(f'bayesian_tests/experiment_21/model/no_calibration/trained_model_{i}.pkl', 'wb') as file:
            pickle.dump((clf), file)

        with open(f'bayesian_tests/experiment_21/model/sigmoid/trained_model_{i}.pkl', 'wb') as file:
            pickle.dump((clf_sigmoid), file)
        
        with open(f'bayesian_tests/experiment_21/model/isotonic/trained_model_{i}.pkl', 'wb') as file:
            pickle.dump((clf_isotonic), file)


        

        
        
        
        test_indicies = X_test_.index

        test_data_df = total_df.loc[test_indicies]
        print(test_data_df)
        test_data_df.to_csv(f'bayesian_tests/experiment_21/test_data/sigmoid_test_data_{i}.csv')
        #print(test_data_df)

        dict_ = {'indicies': test_indicies}
        test_indicies = pd.DataFrame(dict_)
        test_indicies.to_csv(f'bayesian_tests/experiment_21/indicies/indicies_test_{i}.csv')
        print(median(iso_mean))