import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load the training data
df_train = pd.read_csv("Paitients_Files_Train.csv")
df_train['Sepssis'] = df_train['Sepssis'].map({'Positive': 1, 'Negative': 0})

# Split into features and target
X_train = df_train.drop(columns=["ID", "Sepssis"])
y_train = df_train["Sepssis"]

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

# Build the ANN model
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=25, batch_size=35, validation_split=0.3, verbose=1)

# Save the trained model
model.save("sepsis_model.h5")

print("Model and scaler saved successfully!")
