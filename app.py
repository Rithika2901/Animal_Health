import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import kagglehub

# Download the dataset from Kaggle
dataset_path = kagglehub.dataset_download("shijo96john/animal-disease-prediction")
csv_file_path = f"{dataset_path}/cleaned_animal_disease_prediction.csv"
df = pd.read_csv(csv_file_path)

# Extract clusters and disease data
features = df.drop(columns=["Disease_Prediction", "Agglomerative_Cluster"])  # Adjust as per your dataset
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Function to predict cluster for new input
def predict_cluster(user_input):
    user_scaled = scaler.transform([user_input])
    distances = cdist(user_scaled, features_scaled, metric='euclidean')
    closest_index = np.argmin(distances)
    return df.iloc[closest_index]["Agglomerative_Cluster"]

# Streamlit UI
st.title("🐾 Animal Health Tracker & Disease Predictor")

# User Input Form
st.sidebar.header("Enter Animal Information")
animal_type = st.sidebar.selectbox("Animal Type", df["Animal_Type"].unique())
breed = st.sidebar.selectbox("Breed", df["Breed"].unique())
age = st.sidebar.slider("Age", min_value=0, max_value=30, value=5)
gender = st.sidebar.selectbox("Gender", df["Gender"].unique())
weight = st.sidebar.number_input("Weight (kg)", min_value=1.0, value=10.0)

# Symptoms
st.sidebar.header("Select Symptoms")
symptom_1 = st.sidebar.selectbox("Symptom 1", df["Symptom_1"].unique())
symptom_2 = st.sidebar.selectbox("Symptom 2", df["Symptom_2"].unique())
symptom_3 = st.sidebar.selectbox("Symptom 3", df["Symptom_3"].unique())
symptom_4 = st.sidebar.selectbox("Symptom 4", df["Symptom_4"].unique())

# Additional symptoms
appetite_loss = st.sidebar.selectbox("Appetite Loss", ["Yes", "No"])
vomiting = st.sidebar.selectbox("Vomiting", ["Yes", "No"])
diarrhea = st.sidebar.selectbox("Diarrhea", ["Yes", "No"])
coughing = st.sidebar.selectbox("Coughing", ["Yes", "No"])
temperature = st.sidebar.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=38.5)

# Predict Disease
if st.sidebar.button("Predict Disease"):
    user_input = [animal_type, breed, age, gender, weight, symptom_1, symptom_2, symptom_3, symptom_4,
                  appetite_loss, vomiting, diarrhea, coughing, temperature]

    predicted_cluster = predict_cluster(user_input)
    possible_diseases = df[df["Agglomerative_Cluster"] == predicted_cluster]["Disease_Prediction"].value_counts().index.tolist()

    st.subheader(f"📌 Predicted Cluster: {predicted_cluster}")
    st.write(f"🚑 **Possible Diseases:** {', '.join(possible_diseases[:3])}")  # Show top 3 possible diseases
