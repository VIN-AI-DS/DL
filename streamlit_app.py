
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import streamlit as st

# Load the dataset
file_path = '/content/drive/MyDrive/Occupancy_Estimation.csv'  # Adjust this if your file path is different
df = pd.read_csv(file_path)

# Preprocess the data (dropping Date and Time columns)
features = df.drop(columns=['Date', 'Time', 'Room_Occupancy_Count'])
target = df['Room_Occupancy_Count']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model with an explicit Input layer
model = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),  # Explicit input shape
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')  # Linear activation for continuous output
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Make predictions
y_pred_nn = model.predict(X_test_scaled)

# Evaluate the model
mse_nn = mean_squared_error(y_test, y_pred_nn)
r2_nn = r2_score(y_test, y_pred_nn)

st.write(f"Mean Squared Error (MSE): {mse_nn}")
st.write(f"R-squared (R²) Score: {r2_nn}")

# Prediction function that rounds the output to the nearest integer
def predict_occupancy(input_features):
    new_input = np.array([input_features])
    new_input_scaled = scaler.transform(new_input)
    prediction = model.predict(new_input_scaled)
    rounded_prediction = np.round(prediction).astype(int)
    return int(rounded_prediction[0][0])

# Streamlit Interface
st.title("Room Occupancy Estimation")

# Create input fields for each feature
S1_Temp = st.number_input("S1_Temp")
S2_Temp = st.number_input("S2_Temp")
S3_Temp = st.number_input("S3_Temp")
S4_Temp = st.number_input("S4_Temp")
S1_Light = st.number_input("S1_Light")
S2_Light = st.number_input("S2_Light")
S3_Light = st.number_input("S3_Light")
S4_Light = st.number_input("S4_Light")
S1_Sound = st.number_input("S1_Sound")
S2_Sound = st.number_input("S2_Sound")
S3_Sound = st.number_input("S3_Sound")
S4_Sound = st.number_input("S4_Sound")
S5_CO2 = st.number_input("S5_CO2")
S5_CO2_Slope = st.number_input("S5_CO2_Slope")
S6_PIR = st.number_input("S6_PIR")
S7_PIR = st.number_input("S7_PIR")

# Prepare the input features
input_features = [S1_Temp, S2_Temp, S3_Temp, S4_Temp, S1_Light, S2_Light, S3_Light, S4_Light,
                  S1_Sound, S2_Sound, S3_Sound, S4_Sound, S5_CO2, S5_CO2_Slope, S6_PIR, S7_PIR]

# Prediction button
if st.button("Predict Room Occupancy"):
    prediction = predict_occupancy(input_features)
    st.success(f"The predicted room occupancy count is: {prediction}")
