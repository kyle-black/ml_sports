import pymc3 as pm
import numpy as np
import pandas as pd

# Load data
data = pd.read_csv('filled_csv.csv')
data = data[data['home_is_winner'] != 'Unknown']
X = data.drop(['game_id','home_team','away_team','commence_time','id','game_date','season','home_score','away_score','away_is_winner','game_type','home_is_winner', 'betfair_away', 'betfair_home', 'caesars_away', 'caesars_home'], axis=1)
y = data['home_is_winner'].astype(int)

# Standardize the features
X = (X - X.mean()) / X.std()

# Add intercept to X
X = np.hstack([np.ones((X.shape[0], 1)), X])

with pm.Model() as model:
    # Prior distributions for the coefficients
    coeffs = pm.Normal('coeffs', mu=0, sd=1, shape=X.shape[1])

    # Logistic regression model
    p = pm.Deterministic('p', pm.math.sigmoid(pm.math.dot(X, coeffs)))

    # Likelihood
    obs = pm.Bernoulli('obs', p=p, observed=y)

    # Run MCMC
    trace = pm.sample(2000, tune=1000)