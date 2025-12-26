import streamlit as st
import numpy as np
import pandas as pd
import datetime
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from keras.preprocessing import image
import matplotlib.pyplot as plt

# -------------------------------
# Custom CSS for background & styling
# -------------------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #74ebd5 0%, #ACB6E5 100%);
    color: #000000;
}
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
}
h1, h2, h3 {
    font-family: 'Segoe UI', sans-serif;
    font-weight: 600;
    color: #1a1a1a;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    font-size: 16px;
    padding: 8px 16px;
}
.stDownloadButton>button {
    background-color: #2196F3;
    color: white;
    border-radius: 8px;
    font-size: 16px;
    padding: 8px 16px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------------------
# Initialize hydration log storage
# -------------------------------
if "hydration_log" not in st.session_state:
    st.session_state["hydration_log"] = []

# -------------------------------
# Functions for each page
# -------------------------------
def hydration_tracker():
    st.title("💧 Hydration Tracker")

    # Sidebar inputs
    weight = st.sidebar.number_input("Enter your weight (kg)", min_value=30, max_value=200, step=1)
    age = st.sidebar.number_input("Enter your age", min_value=5, max_value=100, step=1)
    gender = st.sidebar.selectbox("Select your gender", ["Male", "Female", "Other"])
    climate = st.sidebar.selectbox("Climate condition", ["Cold", "Moderate", "Hot"])

    # Daily goal suggestion
    base_goal = 2000
    if gender == "Male":
        base_goal += 500
    if climate == "Hot":
        base_goal += 1000
    if weight > 80:
        base_goal += 500
    if age < 18:
        base_goal -= 200

    goal = st.sidebar.number_input("Set daily goal (ml)", min_value=500, value=base_goal, step=100)

    # Intake logging
    amount = st.number_input("Enter water intake (ml)", min_value=100, step=50)
    if st.button("Log Intake"):
        st.session_state["hydration_log"].append({
            "amount": amount,
            "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "weight": weight,
            "age": age,
            "gender": gender,
            "climate": climate
        })
        st.success(f"Logged {amount} ml of water!")

    if st.button("Reset Progress"):
        st.session_state["hydration_log"] = []
        st.warning("Hydration log has been reset!")

    # Display log
    st.write("### Today's Log")
    if st.session_state["hydration_log"]:
        df = pd.DataFrame(st.session_state["hydration_log"])
        st.table(df)

        total = df["amount"].sum()
        avg = df["amount"].mean()

        st.metric("Total Intake (ml)", total)
        st.metric("Average per entry (ml)", f"{avg:.2f}")
        st.metric("Remaining (ml)", max(goal - total, 0))

        st.progress(min(total / goal, 1.0))
    else:
        st.info("No intake logged yet.")

@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

def image_analysis():
    st.title("🖼️ Image Analysis with MobileNetV2")
    model = load_model()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        try:
            preds = model.predict(x)
            decoded = decode_predictions(preds, top=3)[0]

            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            st.write("### Top Predictions")
            for i, (_, label, prob) in enumerate(decoded):
                st.write(f"{i+1}. {label} ({prob*100:.2f}%)")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

def hydration_analysis():
    st.title("📊 Hydration Analysis")

    if st.session_state["hydration_log"]:
        df = pd.DataFrame(st.session_state["hydration_log"])

        st.write("### Intake Summary")
        total = df["amount"].sum()
        avg = df["amount"].mean()
        st.metric("Total Intake (ml)", total)
        st.metric("Average per entry (ml)", f"{avg:.2f}")

        # Bar chart
        st.bar_chart(df["amount"])

        # Line chart with time
        df["time"] = pd.to_datetime(df["time"])
        st.line_chart(df.set_index("time")["amount"])

        # Download option
        st.download_button("Download Log", df.to_csv(index=False), "hydration_log.csv")
    else:
        st.info("No hydration data to analyze yet.")

# -------------------------------
# Sidebar navigation
# -------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["💧 Hydration Tracker", "🖼️ Image Analysis", "📊 Hydration Analysis"])

if page == "💧 Hydration Tracker":
    hydration_tracker()
elif page == "🖼️ Image Analysis":
    image_analysis()
elif page == "📊 Hydration Analysis":
    hydration_analysis()
