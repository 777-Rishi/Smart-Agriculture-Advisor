import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# =============================
# FILE PATHS
# =============================
CROP_DATA_PATH = r"C:\Users\Admin\OneDrive\ドキュメント\datasets\crop_recommendation_dataset.csv"
YIELD_DATA_PATH = r"C:\Users\Admin\OneDrive\ドキュメント\datasets\crop yeild\crop_yield.csv"

CROP_MODEL_FILE = "crop_model.pkl"
SOIL_ENCODER_FILE = "soil_encoder.pkl"
CROP_LABEL_ENCODER_FILE = "crop_label_encoder.pkl"

YIELD_MODEL_FILE = "yield_model.pkl"
YIELD_CROP_ENCODER_FILE = "yield_crop_encoder.pkl"
SEASON_ENCODER_FILE = "season_encoder.pkl"
STATE_ENCODER_FILE = "state_encoder.pkl"

# =============================
# TRAINING FUNCTION
# =============================
def train_models():
    # -----------------
    # Crop Recommendation Model
    # -----------------
    crop_df = pd.read_csv(CROP_DATA_PATH)
    soil_encoder = LabelEncoder()
    crop_label_encoder = LabelEncoder()

    crop_df["Soil"] = soil_encoder.fit_transform(crop_df["Soil"])
    crop_df["Crop"] = crop_label_encoder.fit_transform(crop_df["Crop"])

    X_crop = crop_df.drop("Crop", axis=1)
    y_crop = crop_df["Crop"]

    X_train, X_test, y_train, y_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)
    crop_model = RandomForestClassifier(random_state=42)
    crop_model.fit(X_train, y_train)
    acc = accuracy_score(y_test, crop_model.predict(X_test))
    print(f"Crop Recommendation Model Accuracy: {acc:.2f}")

    with open(CROP_MODEL_FILE, "wb") as f:
        pickle.dump(crop_model, f)
    with open(SOIL_ENCODER_FILE, "wb") as f:
        pickle.dump(soil_encoder, f)
    with open(CROP_LABEL_ENCODER_FILE, "wb") as f:
        pickle.dump(crop_label_encoder, f)

    # -----------------
    # Yield Prediction Model
    # -----------------
    yield_df = pd.read_csv(YIELD_DATA_PATH)
    yield_crop_encoder = LabelEncoder()
    season_encoder = LabelEncoder()
    state_encoder = LabelEncoder()

    yield_df["Crop"] = yield_crop_encoder.fit_transform(yield_df["Crop"])
    yield_df["Season"] = season_encoder.fit_transform(yield_df["Season"])
    yield_df["State"] = state_encoder.fit_transform(yield_df["State"])

    X_yield = yield_df.drop("Yield", axis=1)
    y_yield = yield_df["Yield"]

    X_train, X_test, y_train, y_test = train_test_split(X_yield, y_yield, test_size=0.2, random_state=42)
    yield_model = RandomForestRegressor(random_state=42)
    yield_model.fit(X_train, y_train)
    mse = mean_squared_error(y_test, yield_model.predict(X_test))
    print(f"Yield Prediction Model MSE: {mse:.2f}")

    with open(YIELD_MODEL_FILE, "wb") as f:
        pickle.dump(yield_model, f)
    with open(YIELD_CROP_ENCODER_FILE, "wb") as f:
        pickle.dump(yield_crop_encoder, f)
    with open(SEASON_ENCODER_FILE, "wb") as f:
        pickle.dump(season_encoder, f)
    with open(STATE_ENCODER_FILE, "wb") as f:
        pickle.dump(state_encoder, f)

# =============================
# LOAD OR RETRAIN MODELS
# =============================
def safe_load_model(file_path):
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        st.warning(f"⚠ Model/encoder file '{file_path}' is missing or corrupted. Retraining models...")
        train_models()
        with open(file_path, "rb") as f:
            return pickle.load(f)

# =============================
# MAIN STREAMLIT APP
# =============================
st.set_page_config(page_title="Crop & Yield Prediction", layout="wide")
st.title("🌾 Smart Agriculture Advisor")

tab1, tab2 = st.tabs(["🌱 Crop Recommendation", "📊 Yield Prediction"])

# -----------------
# Crop Recommendation Tab
# -----------------
with tab1:
    st.header("🌱 Crop Recommendation System")

    crop_model = safe_load_model(CROP_MODEL_FILE)
    soil_encoder = safe_load_model(SOIL_ENCODER_FILE)
    crop_label_encoder = safe_load_model(CROP_LABEL_ENCODER_FILE)

    df_crop = pd.read_csv(CROP_DATA_PATH)
    soil_types = sorted(df_crop["Soil"].unique())

    col1, col2, col3 = st.columns(3)
    with col1:
        temp = st.number_input("🌡 Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
        humidity = st.number_input("💧 Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
        rainfall = st.number_input("🌧 Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)
    with col2:
        ph = st.number_input("⚖ PH", min_value=0.0, max_value=14.0, value=6.5)
        nitrogen = st.number_input("🧪 Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0)
        phosphorous = st.number_input("🧪 Phosphorous (P)", min_value=0.0, max_value=200.0, value=50.0)
    with col3:
        potassium = st.number_input("🧪 Potassium (K)", min_value=0.0, max_value=200.0, value=50.0)
        carbon = st.number_input("🌱 Carbon (%)", min_value=0.0, max_value=10.0, value=2.0)
        soil = st.selectbox("🪨 Soil Type", soil_types)

    if st.button("🔍 Recommend Crop"):
        soil_encoded = soil_encoder.transform([soil])[0]
        features = [[temp, humidity, rainfall, ph, nitrogen, phosphorous, potassium, carbon, soil_encoded]]
        prediction = crop_model.predict(features)[0]
        crop_name = crop_label_encoder.inverse_transform([prediction])[0]
        st.success(f"✅ Recommended Crop: *{crop_name}*")

# -----------------
# Yield Prediction Tab
# -----------------
with tab2:
    st.header("📊 Crop Yield Prediction")

    yield_model = safe_load_model(YIELD_MODEL_FILE)
    yield_crop_encoder = safe_load_model(YIELD_CROP_ENCODER_FILE)
    season_encoder = safe_load_model(SEASON_ENCODER_FILE)
    state_encoder = safe_load_model(STATE_ENCODER_FILE)

    df_yield = pd.read_csv(YIELD_DATA_PATH)
    crop_options = sorted(df_yield["Crop"].unique())
    season_options = sorted(df_yield["Season"].unique())
    state_options = sorted(df_yield["State"].unique())

    col1, col2, col3 = st.columns(3)
    with col1:
        crop_choice = st.selectbox("🌾 Crop", crop_options)
        crop_year = st.number_input("📅 Crop Year", min_value=2000, max_value=2030, value=2022)
        season_choice = st.selectbox("🗓 Season", season_options)
    with col2:
        state_choice = st.selectbox("🏛 State", state_options)
        area = st.number_input("📐 Area (Hectares)", min_value=0.1, value=1.0)
        production = st.number_input("🏭 Production (Tonnes)", min_value=0.0, value=1.0)
    with col3:
        annual_rainfall = st.number_input("🌧 Annual Rainfall (mm)", min_value=0.0, value=1000.0)
        fertilizer = st.number_input("🧪 Fertilizer (Kg)", min_value=0.0, value=50.0)
        pesticide = st.number_input("🧪 Pesticide (Kg)", min_value=0.0, value=5.0)

    if st.button("📈 Predict Yield"):
        crop_encoded = yield_crop_encoder.transform([crop_choice])[0]
        season_encoded = season_encoder.transform([season_choice])[0]
        state_encoded = state_encoder.transform([state_choice])[0]
        features = [[crop_encoded, crop_year, season_encoded, state_encoded, area,
                     production, annual_rainfall, fertilizer, pesticide]]
        prediction = yield_model.predict(features)[0]
        st.success(f"🌾 Predicted Yield: *{prediction:.2f} tonnes/hectare*")