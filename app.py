import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import kagglehub

# ✅ Download the dataset from Kaggle
dataset_path = kagglehub.dataset_download("shijo96john/animal-disease-prediction")
csv_file_path = f"{dataset_path}/cleaned_animal_disease_prediction.csv"
df = pd.read_csv(csv_file_path)

# ✅ Print column names to debug missing columns
st.write("Columns in dataset:", df.columns.tolist())

# ✅ Check if columns exist before dropping them
columns_to_drop = ["Disease_Prediction", "Agglomerative_Cluster"]
existing_columns = [col for col in columns_to_drop if col in df.columns]

if existing_columns:
    features = df.drop(columns=existing_columns)  # Drop only existing columns
else:
    features = df.copy()  # If columns don't exist, use full dataset

# ✅ Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ✅ Function to predict cluster for new input
def predict_cluster(user_input):
    user_scaled = scaler.transform([user_input])
    distances = cdist(user_scaled, features_scaled, metric='euclidean')
    closest_index = np.argmin(distances)
    return df.iloc[closest_index]["Agglomerative_Cluster"] if "Agglomerative_Cluster" in df.columns else "Unknown"

# ✅ Streamlit UI
st.title("🐾 Animal Health Tracker & Disease Predictor")

# ✅ User Input Form
st.sidebar.header("Enter Animal Information")
animal_type = st.sidebar.selectbox("Animal Type", df["Animal_Type"].unique())
breed = st.sidebar.selectbox("Breed", df["Breed"].unique())
age = st.sidebar.slider("Age", min_value=0, max_value=30, value=5)
gender = st.sidebar.selectbox("Gender", df["Gender"].unique())
weight = st.sidebar.number_input("Weight (kg)", min_value=1.0, value=10.0)

# ✅ Symptoms
st.sidebar.header("Select Symptoms")
symptom_1 = st.sidebar.selectbox("Symptom 1", df["Symptom_1"].unique())
symptom_2 = st.sidebar.selectbox("Symptom 2", df["Symptom_2"].unique())
symptom_3 = st.sidebar.selectbox("Symptom 3", df["Symptom_3"].unique())
symptom_4 = st.sidebar.selectbox("Symptom 4", df["Symptom_4"].unique())

# ✅ Additional symptoms
appetite_loss = st.sidebar.selectbox("Appetite Loss", ["Yes", "No"])
vomiting = st.sidebar.selectbox("Vomiting", ["Yes", "No"])
diarrhea = st.sidebar.selectbox("Diarrhea", ["Yes", "No"])
coughing = st.sidebar.selectbox("Coughing", ["Yes", "No"])
temperature = st.sidebar.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=38.5)

# ✅ Predict Disease
if st.sidebar.button("Predict Disease"):
    user_input = [animal_type, breed, age, gender, weight, symptom_1, symptom_2, symptom_3, symptom_4,
                  appetite_loss, vomiting, diarrhea, coughing, temperature]

    predicted_cluster = predict_cluster(user_input)

    # ✅ Ensure Disease_Prediction exists before using it
    if "Disease_Prediction" in df.columns and "Agglomerative_Cluster" in df.columns:
        possible_diseases = df[df["Agglomerative_Cluster"] == predicted_cluster]["Disease_Prediction"].value_counts().index.tolist()
        disease_list = ", ".join(possible_diseases[:3]) if possible_diseases else "No data available"
    else:
        disease_list = "No disease data available"

    st.subheader(f"📌 Predicted Cluster: {predicted_cluster}")
    st.write(f"🚑 **Possible Diseases:** {disease_list}")
