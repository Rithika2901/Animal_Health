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

# ✅ Print column names to debug missing columns
st.write("Columns in dataset:", df.columns.tolist())

# ✅ Identify categorical columns
categorical_cols = [
    "Animal_Type", "Breed", "Gender", "Symptom_1", "Symptom_2", "Symptom_3", "Symptom_4",
    "Duration", "Appetite_Loss", "Vomiting", "Diarrhea", "Coughing", "Labored_Breathing",
    "Lameness", "Skin_Lesions", "Nasal_Discharge", "Eye_Discharge", "Body_Temperature"
]

# ✅ Encode categorical columns into numeric values
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders if needed later

# ✅ Drop `Disease_Prediction` column (as it's a text column and not needed for clustering)
features = df.drop(columns=["Disease_Prediction"]) 

# ✅ Standardize numerical features
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

# ✅ Convert user input into numeric format
user_input = [
    label_encoders["Animal_Type"].transform([animal_type])[0],
    label_encoders["Breed"].transform([breed])[0],
    age,
    label_encoders["Gender"].transform([gender])[0],
    weight,
    label_encoders["Symptom_1"].transform([symptom_1])[0],
    label_encoders["Symptom_2"].transform([symptom_2])[0],
    label_encoders["Symptom_3"].transform([symptom_3])[0],
    label_encoders["Symptom_4"].transform([symptom_4])[0],
    1 if appetite_loss == "Yes" else 0,
    1 if vomiting == "Yes" else 0,
    1 if diarrhea == "Yes" else 0,
    1 if coughing == "Yes" else 0,
    temperature
]

# ✅ Predict Disease
if st.sidebar.button("Predict Disease"):
    predicted_cluster = predict_cluster(user_input)

    # ✅ Ensure `Disease_Prediction` exists before using it
    if "Disease_Prediction" in df.columns and "Agglomerative_Cluster" in df.columns:
        possible_diseases = df[df["Agglomerative_Cluster"] == predicted_cluster]["Disease_Prediction"].value_counts().index.tolist()
        disease_list = ", ".join(possible_diseases[:3]) if possible_diseases else "No data available"
    else:
        disease_list = "No disease data available"

    st.subheader(f"📌 Predicted Cluster: {predicted_cluster}")
    st.write(f"🚑 **Possible Diseases:** {disease_list}")

