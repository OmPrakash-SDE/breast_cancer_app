import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset and train model
@st.cache_resource
def load_model():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)
    
    # Assign actual column names
    columns = ['ID', 'Diagnosis', 
               'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean',
               'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
               'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se',
               'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
               'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst',
               'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']
    df.columns = columns

    # Drop ID column
    df = df.drop(columns=['ID'])

    # Convert Diagnosis column to binary (M = 1, B = 0)
    df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})

    # Split dataset into features and target
    X = df.drop(columns=['Diagnosis'])
    y = df['Diagnosis']

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression model
    model = LogisticRegression()
    model.fit(X_scaled, y)

    return model, scaler

# Load trained model
model, scaler = load_model()

# Streamlit App UI
st.title("Breast Cancer Prediction App")
st.write("Enter the values below to predict whether the tumor is **Benign** or **Malignant**.")

# Feature names
feature_names = [
    'Radius Mean', 'Texture Mean', 'Perimeter Mean', 'Area Mean', 'Smoothness Mean', 'Compactness Mean',
    'Concavity Mean', 'Concave Points Mean', 'Symmetry Mean', 'Fractal Dimension Mean',
    'Radius SE', 'Texture SE', 'Perimeter SE', 'Area SE', 'Smoothness SE', 'Compactness SE',
    'Concavity SE', 'Concave Points SE', 'Symmetry SE', 'Fractal Dimension SE',
    'Radius Worst', 'Texture Worst', 'Perimeter Worst', 'Area Worst', 'Smoothness Worst', 'Compactness Worst',
    'Concavity Worst', 'Concave Points Worst', 'Symmetry Worst', 'Fractal Dimension Worst'
]

# Create input fields for each feature
input_features = []
for feature in feature_names:
    value = st.number_input(f'{feature}', min_value=0.0, step=0.01, format="%.4f")
    input_features.append(value)

# Predict button
if st.button("Predict"):
    input_array = np.array([input_features]).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    
    # Display Result
    if prediction[0] == 1:
        st.error("Prediction: Malignant (Cancerous)")
    else:
        st.success("Prediction: Benign (Non-Cancerous)")
