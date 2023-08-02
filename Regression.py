from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from statistics import mean

from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from collections import defaultdict
from sklearn.metrics import log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns



def log_training(data, y, cols):
    # Split the data into training and test sets
    brier_score_dict = defaultdict(list)
    log_loss_dict = defaultdict(list)
    calibration_data_dict = defaultdict(lambda: defaultdict(list))  # For storing predicted probabilities and true labels
    prediction_distribution_data = defaultdict(list)  # For storing predictions for distribution plots

    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(data[cols], y, test_size=0.2, random_state=i)

        log_reg = LogisticRegression(warm_start=True)
        log_reg.fit(X_train, y_train)
        
        # Save the model to a file
        joblib.dump(log_reg, 'log_reg_model.pkl')  # change the path if necessary
        
        # Predict the probabilities for the test set
        y_proba = log_reg.predict_proba(X_test)
        
        # Predict the classes for the test set
        y_pred = log_reg.predict(X_test)
        
        # Compute accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy:', accuracy)

        # Compute ROC AUC
        roc_auc = roc_auc_score(y_test, y_proba[:, 1])
        print('ROC AUC:', roc_auc)

        # Compute Brier Score and Log Loss for model
        brier_score1 = brier_score_loss(y_test, y_proba[:, 1])
        print('Brier_Score:', brier_score1)
        log_loss1 = log_loss(y_test, y_proba)
        print('Log Loss:', log_loss1)

        # Compute Brier Score and Log Loss for each column that ends with '_home_vf'
        for col in X_test.filter(like='_home_prob', axis=1).columns:
            try:
                brier_score = brier_score_loss(y_test, X_test[col])
                log_loss_score = log_loss(y_test, X_test[col])
                brier_score_dict[col].append(brier_score)
                log_loss_dict[col].append(log_loss_score)
                print(f'Brier Score for {col}:', brier_score)
                print(f'Log Loss for {col}:', log_loss_score)

                # Store predicted probabilities and true labels for calibration curves
                calibration_data_dict[col]["y_true"].extend(y_test)
                calibration_data_dict[col]["y_pred"].extend(X_test[col])
                
                # Store predicted probabilities for distribution plots
                prediction_distribution_data[col].extend(X_test[col])

            except ValueError as e:
                print(f"Error calculating metrics for {col}: {e}")

    # Calculate and print mean Brier scores and Log Loss for each book
    for book, scores in brier_score_dict.items():
        print(f'Mean Brier score for {book}: {sum(scores)/len(scores)}')
        print(f'Mean Log Loss for {book}: {sum(log_loss_dict[book])/len(log_loss_dict[book])}')

    # Plot calibration curves for each book
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    for book, data in calibration_data_dict.items():
        fraction_of_positives, mean_predicted_value = calibration_curve(data["y_true"], data["y_pred"], n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, label=book)
    plt.ylabel('Fraction of positives')
    plt.xlabel('Average predicted value')
    plt.legend(loc='lower right')
    plt.title('Calibration curves')
    
    # Plot distribution of predicted probabilities
    plt.figure(figsize=(10, 10))
    for book, predictions in prediction_distribution_data.items():
        sns.histplot(predictions, bins=30, kde=True, label=book)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title('Distribution of Predicted Probabilities')
    
    plt.show()


def linear_training(data, y, cols):
    # Split the data into training and test sets
    mse_list =[]
    r2_list = []
    mae_list =[]
    for i in range(100):
        X_train, X_test, y_train, y_test = train_test_split(data[cols], y, test_size=0.2, random_state=i)

        lin_reg = LinearRegression()
        lin_reg.fit(X_train, y_train)
        
        # Save the model to a file
        joblib.dump(lin_reg, 'lin_reg_model.pkl')  # change the path if necessary
        
        # Predict the output for the test set
        y_pred = lin_reg.predict(X_test)
        
        # Compute mean squared error
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse_list.append(mse)
        print('Mean Squared Error:', mse)

        # Compute R^2 Score
        r2 = r2_score(y_test, y_pred)
        r2_list.append(r2)
        mae_list.append(mae)
        print('R^2 Score:', r2)

        # Compute Brier Score for each column that ends with '_home_vf'
        

    print('Mean of Mean Squared Errors:', mean(mse_list)) 
    print('Mean of R^2 Scores:', mean(r2_list))
    print('Mean of Mean absolute Errors:', mean(mae_list))  