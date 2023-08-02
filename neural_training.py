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

import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp

def bayesian_neural_network(data, y, cols):
    # Split the data into training and test sets
    brier_score_dict = defaultdict(list)
    calibration_data_dict = defaultdict(lambda: defaultdict(list))

    #neg_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)
    #for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(data[cols], y, test_size=0.2, random_state=42)

        # Define Bayesian Neural Network with keras and tfp
    model = keras.models.Sequential([
        tfp.layers.DenseFlipout(30, activation='relu', input_shape=X_train.shape[1:]),
        tfp.layers.DenseFlipout(30, activation='relu'),
        tfp.layers.DenseFlipout(30, activation='relu'),
        tfp.layers.DenseFlipout(1, activation='sigmoid')
])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
              optimizer="adam", 
              metrics=["binary_crossentropy", "MeanSquaredError"])

    model.fit(X_train, y_train, epochs=50, verbose=1)

        # Predict the probabilities for the test set
    y_proba = model.predict(X_test)
    y_pred = (y_proba > 0.5).astype(int)  # Converting probabilities to class labels

        # Compute accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

        # Compute ROC AUC
    roc_auc = roc_auc_score(y_test, y_proba)
    print('ROC AUC:', roc_auc)

        # Compute Brier Score for model
    brier_score1 = brier_score_loss(y_test, y_proba)
    print('Brier_Score:', brier_score1)

        # Compute Brier Score for each column that ends with '_home_vf'
    for col in X_test.filter(like='_home_prob', axis=1).columns:
        try:
            brier_score = brier_score_loss(y_test, X_test[col])
            brier_score_dict[col].append(brier_score)
            print(f'Brier Score for {col}:', brier_score)

                # Store predicted probabilities and true labels for calibration curves
            calibration_data_dict[col]["y_true"].extend(y_test)
            calibration_data_dict[col]["y_pred"].extend(X_test[col])

        except ValueError as e:
            print(f"Error calculating metrics for {col}: {e}")

    # Calculate and print mean Brier scores for each book
    for book, scores in brier_score_dict.items():
        print(f'Mean Brier score for {book}: {sum(scores)/len(scores)}')

    mean_brier_scores = {book: mean(scores) for book, scores in brier_score_dict.items()}

# Prepare data for plotting
    books = list(mean_brier_scores.keys())
    scores = list(mean_brier_scores.values())

# Create a bar chart
    plt.figure(figsize=[10,6])
    plt.bar(books, scores)
    plt.xlabel('Sports Book')
    plt.ylabel('Mean Brier Score')
    plt.title('Comparison of Mean Brier Scores among Sports Books')
    plt.show()

    '''
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
    plt.show()
    '''