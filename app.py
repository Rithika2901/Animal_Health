import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.spatial.distance import cdist
import kagglehub

# ✅ Download the dataset from Kaggle
dataset_path = kagglehub.dataset_download("shijo96john/animal-disease-prediction")
csv_file_path = f"{dataset_path}/cleaned_animal_disease_prediction.csv"
df = pd.read_csv(csv_file_path)

# ✅ Print column names (to verify data structure)
st.write("✅ Dataset Loaded! Columns:", df.columns.tolist())

# ✅ Identify categorical columns
categorical_cols = [
    "Animal_Type", "Breed", "Gender", "Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4",
    "Duration", "Appetite_Loss", "Vomiting", "Diarrhea", "Coughing", "Labored_Breathing",
    "Lameness", "Skin_Lesions", "Nasal_Discharge", "Eye_Discharge"
]

# ✅ Convert categorical columns to numeric using LabelEncoder
label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Ensure values are strings before encoding
        label_encoders[col] = le  # Store encoders for future decoding

# ✅ Drop non-numeric columns before scaling
numeric_df = df.select_dtypes(include=[np.number])  # Keep only numeric columns
if "Disease_Prediction" in numeric_df.columns:
    numeric_df = numeric_df.drop(columns=["Disease_Prediction"])  # Drop text-based target column if exists

# ✅ Apply StandardScaler only on numeric data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(numeric_df)

# ✅ Function to predict cluster for new input
def predict_cluster(user_input):
    user_scaled = scaler.transform([user_input])
    distances = cdist(user_scaled, features_scaled, metric='euclidean')
    closest_index = np.argmin(distances)
    return df.iloc[closest_index]["Agglomerative_Cluster"] if "Agglomerative_Cluster" in df.columns else "Unknown"

# 🎯 Streamlit UI
st.title("🚀 Animal Health Tracker & Disease Predictor")
st.markdown("🔬 **Enter details to predict the most likely disease.**")

# ✅ User Input Form
animal_type = st.selectbox("Select Animal Type", df["Animal_Type"].unique())
breed = st.selectbox("Select Breed", df["Breed"].unique())
age = st.slider("Age of the Animal", min_value=0, max_value=30, value=5)
weight = st.number_input("Weight (kg)", min_value=1.0, value=10.0)
temperature = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=38.5)

# ✅ Symptoms Selection
symptoms = st.multiselect("Select Symptoms", df["Symptom_1"].unique())

# ✅ Convert user input into numeric format
user_input = [
    label_encoders["Animal_Type"].transform([animal_type])[0],
    label_encoders["Breed"].transform([breed])[0],
    age,
    weight,
    temperature,
]

for symptom in ["Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4"]:
    user_input.append(label_encoders[symptom].transform([symptoms[0]])[0] if symptoms else 0)

# 🎯 Predict Disease
if st.button("🔍 Predict Disease"):
    predicted_cluster = predict_cluster(user_input)

    if "Disease_Prediction" in df.columns and "Agglomerative_Cluster" in df.columns:
        possible_diseases = df[df["Agglomerative_Cluster"] == predicted_cluster]["Disease_Prediction"].value_counts().index.tolist()
        disease_list = f"**{possible_diseases[0]}**" if possible_diseases else "No data available"
    else:
        disease_list = "No disease data available"

    st.success(f"✅ Based on the given information, the predicted disease is **{disease_list}**.")


