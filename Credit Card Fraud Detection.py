import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv('creditcard.csv')

# Check for missing values
print(df.isnull().sum())

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Build Neural Network
nn_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
nn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
y_pred_nn = nn_model.predict(X_test)
y_pred_nn = (y_pred_nn > 0.5).astype(int)

print(confusion_matrix(y_test, y_pred_nn))
print(classification_report(y_test, y_pred_nn))

pip install Flask
from flask import Flask, request, jsonify
import numpy as np
import joblib

# Load the trained models
rf_model = joblib.load('rf_model.pkl')
nn_model = tf.keras.models.load_model('nn_model.h5')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    
    # Choose the model to use
    model_choice = data.get('model', 'random_forest')
    
    if model_choice == 'neural_network':
        features = scaler.transform(features)  # Ensure the features are scaled
        prediction = nn_model.predict(features)
        prediction = (prediction > 0.5).astype(int)[0][0]
    else:
        prediction = rf_model.predict(features)[0]

    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
import joblib
joblib.dump(rf_model, 'rf_model.pkl')
nn_model.save('nn_model.h5')

python app.py

