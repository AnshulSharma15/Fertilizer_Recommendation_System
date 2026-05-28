
import streamlit as st
import pickle
import numpy as np

# Load model and encoders
model = pickle.load(open("fertilizer_model.pkl", "rb"))
soil_encoder = pickle.load(open("soil_encoder.pkl", "rb"))
crop_encoder = pickle.load(open("crop_encoder.pkl", "rb"))
fert_encoder = pickle.load(open("fert_encoder.pkl", "rb"))

# Page settings
st.set_page_config(
    page_title="Fertilizer Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# CSS
st.markdown("""
<style>

.stApp {
    background-image:
    linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)),
    url("https://images.unsplash.com/photo-1464226184884-fa280b87c399");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.block-container {
    padding-top: 2rem;
    padding-left: 4rem;
    padding-right: 4rem;
}

.title {
    text-align: center;
    font-size: 60px;
    font-weight: bold;
    color: #76ff03;
}

.subtitle {
    text-align: center;
    color: white;
    font-size: 22px;
    margin-bottom: 40px;
}

.stNumberInput label,
.stSelectbox label {
    color: white !important;
    font-size: 18px !important;
    font-weight: bold;
}

div[data-baseweb="input"] {
    background-color: rgba(255,255,255,0.08);
    border-radius: 12px;
}

div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.08);
    border-radius: 12px;
}

.stButton>button {
    width: 100%;
    height: 70px;
    border-radius: 15px;
    border: none;
    background: linear-gradient(to right, #00c853, #64dd17);
    color: white;
    font-size: 28px;
    font-weight: bold;
    margin-top: 30px;
    box-shadow: 0px 0px 20px rgba(0,255,100,0.4);
}

.result {
    margin-top: 40px;
    background: rgba(0,0,0,0.5);
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    border: 1px solid rgba(0,255,100,0.3);
}

.result h2 {
    color: white;
    font-size: 30px;
}

.result h1 {
    color: #76ff03;
    font-size: 55px;
}

.confidence {
    color: white;
    font-size: 22px;
    margin-top: 10px;
}
.stSelectbox div[data-baseweb="select"] > div {
    background-color: rgba(255,255,255,0.15);
    color: white !important;
    border-radius: 10px;
}

.stSelectbox svg {
    fill: white !important;
}

.stSelectbox option {
    color: black !important;
}

</style>
""", unsafe_allow_html=True)

# Title
st.markdown(
    '<div class="title">🌱 Fertilizer Recommendation System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI Powered Smart Farming Assistant</div>',
    unsafe_allow_html=True
)

# Columns
col1, col2 = st.columns(2)

with col1:

    temperature = st.number_input(
        "🌡 Temperature (°C)",
        min_value=0.0,
        max_value=50.0,
        value=25.0,
        step=0.1
    )

    humidity = st.number_input(
        "💧 Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.1
    )

    soil_moisture = st.number_input(
        "🌱 Soil Moisture (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=0.1
    )

    nitrogen = st.number_input(
        "🧪 Nitrogen",
        min_value=0,
        max_value=140,
        value=20,
        step=1
    )

with col2:

    potassium = st.number_input(
        "⚡ Potassium",
        min_value=0,
        max_value=205,
        value=20,
        step=1
    )

    phosphorous = st.number_input(
        "🧬 Phosphorous",
        min_value=0,
        max_value=145,
        value=20,
        step=1
    )

    soil_type = st.selectbox(
        "🪨 Soil Type",
        soil_encoder.classes_
    )

    crop_type = st.selectbox(
        "🌾 Crop Type",
        crop_encoder.classes_
    )
# Button
if st.button("🚀 Predict Best Fertilizer"):

    soil_encoded = soil_encoder.transform([soil_type])[0]
    crop_encoded = crop_encoder.transform([crop_type])[0]

    data = np.array([[
        temperature,
        humidity,
        soil_moisture,
        soil_encoded,
        crop_encoded,
        nitrogen,
        potassium,
        phosphorous
    ]])

    prediction = model.predict(data)[0]

    fertilizer = fert_encoder.inverse_transform([prediction])[0]

    confidence = np.max(model.predict_proba(data)) * 100

    st.markdown(f"""
    <div class="result">
        <h2>🧪 Recommended Fertilizer</h2>
        <h1>{fertilizer.upper()}</h1>
        <div class="confidence">
            Confidence Score: {confidence:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)