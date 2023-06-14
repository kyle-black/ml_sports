import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load and preprocess your dataset
data = pd.read_csv('19_21_22_results.csv')

# Define X and y variables
X = data.drop(['home_is_winner', 'game_date', 'home_tm', 'away_tm'], axis=1)
y = data['home_is_winner']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define a function that creates and compiles a model with given hyperparameters
def create_model(learning_rate, num_hidden_layers, num_units):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    
    for _ in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(num_units, activation='relu'))
    
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Define your hyperparameter search space
learning_rates = [0.001, 0.01]
num_hidden_layers_list = [1, 2]
num_units_list = [16, 32]

best_val_accuracy = 0
best_hyperparams = None

# Iterate through the hyperparameter combinations
for learning_rate in learning_rates:
    for num_hidden_layers in num_hidden_layers_list:
        for num_units in num_units_list:
            model = create_model(learning_rate, num_hidden_layers, num_units)
            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
            _, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
            
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_hyperparams = (learning_rate, num_hidden_layers, num_units)

# Train the model using the best hyperparameters and the entire dataset
learning_rate, num_hidden_layers, num_units = best_hyperparams
model = create_model(learning_rate, num_hidden_layers, num_units)
model.fit(X_scaled, y, epochs=50, batch_size=32, verbose=0)

# Save the model
model.save('best_model.h5')