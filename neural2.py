from tensorflow_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

# Load and preprocess your dataset
data = pd.read_csv('filled_csv.csv')
data = data[data['home_is_winner'] != 'Unknown']

# Assume that all columns are features except the last one, which is the target
X = data.drop(['game_id','home_team','away_team','commence_time','id','game_date','season','home_score','away_score','away_is_winner','game_type','home_is_winner', 'betfair_away', 'betfair_home', 'caesars_away', 'caesars_home'], axis=1).values
y = LabelEncoder().fit_transform(data['home_is_winner'].astype(int))

# Perform a train-test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TabNet classifier
clf = TabNetClassifier()

# Train the classifier
clf.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    eval_metric=['auc'],
    max_epochs=100,
    batch_size=1024,
    virtual_batch_size=128,
    patience=10  # stop training if the validation score does not improve for 10 consecutive epochs
)

# Predict on the validation set
preds_proba = clf.predict_proba(X_valid)

# Calculate Brier Score Loss
brier_loss = brier_score_loss(y_valid, preds_proba[:, 1])
print('Brier Score Loss:', brier_loss)

# Calculate Log Loss
logloss = log_loss(y_valid, preds_proba)
print('Log Loss:', logloss)

# Predict classes
preds_class = clf.predict(X_valid)

# Calculate the accuracy
accuracy = accuracy_score(y_valid, preds_class)
print('Validation accuracy:', accuracy)