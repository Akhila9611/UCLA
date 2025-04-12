import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from src.features.build_features import preprocess_data
from src.models.predict_model import predict
import pickle

# Load the pickled model
model_path = 'src/models/mlp_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load and preprocess data (same as before)
data = pd.read_csv('data/raw/Admission.csv')
data['Admit_Chance'] = (data['Admit_Chance'] >= 0.8).astype(int)
X, y = preprocess_data(data)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Streamlit UI
st.title("Predicting Admission Chances at UCLA")

# Get user input for features
GRE_Score = st.number_input("GRE Score", min_value=0, max_value=340, value=320)
TOEFL_Score = st.number_input("TOEFL Score", min_value=0, max_value=120, value=105)
University_Rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])
SOP = st.selectbox("Statement of Purpose Strength (1-5)", [1, 2, 3, 4, 5])
LOR = st.selectbox("Letter of Recommendation Strength (1-5)", [1, 2, 3, 4, 5])
CGPA = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.5)
Research = st.selectbox("Research Experience", ["Yes", "No"])

# Prepare input data
user_input = pd.DataFrame({
    'GRE_Score': [GRE_Score],
    'TOEFL_Score': [TOEFL_Score],
    'University_Rating': [University_Rating],
    'SOP': [SOP],
    'LOR': [LOR],
    'CGPA': [CGPA],
    'Research': [1 if Research == "Yes" else 0]
})

# Align the user input columns with training data columns (create dummies if needed)
def align_columns(X_train, input_df):
    """
    Align the input data columns with the training data columns.
    If any columns are missing in the input, add them with default values (0).
    Also ensure that categorical columns are one-hot encoded.
    """
    train_columns = X_train.columns
    
    # One-hot encode categorical columns for the user input
    input_df = pd.get_dummies(input_df, columns=['University_Rating', 'Research'], dtype='int')
    
    # Ensure that input_df has the same columns as the training data
    for col in train_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing column with default value (0)

    # Reorder columns to match the training data
    input_df = input_df[train_columns]

    return input_df

# Align the columns of the user input to match the training data
input_df = align_columns(X_train, user_input)

# Scale the input data using the same scaler as the training data
input_scaled = scaler.transform(input_df)

# Set a custom threshold for prediction
threshold = 0.4  # Adjusted threshold to allow more flexibility for predictions

# Add a submit button to trigger prediction
if st.button('Submit'):
    # Get prediction probabilities
    prediction_proba = model.predict_proba(input_scaled)[:, 1]  # Probability of being admitted

    # Debugging: Check the prediction probabilities
    st.write(f"Admission Probability: {prediction_proba[0] * 100:.2f}%")

    # Make prediction based on threshold
    prediction = 1 if prediction_proba[0] >= threshold else 0
    st.write(f"Prediction: {'Admitted' if prediction == 1 else 'Not Admitted'}")
