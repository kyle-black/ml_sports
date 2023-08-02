import pandas as pd
import numpy as np
import sqlite3
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.lines as line
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC


def measurement_(df):
    df = df.loc[df['home_is_winner'] != 'Unknown'].copy()  # Use .copy() to make an independent copy
    y_true = df['home_is_winner'].astype(int)

    for col in df.filter(like='_home_vf', axis=1).columns:
        
        try:
            df.loc[:, f'{col}_score'] = brier_score_loss(y_true, df[col])
        except ValueError as e:
            print(f"Error calculating Brier score for {col}: {e}")

    return df


