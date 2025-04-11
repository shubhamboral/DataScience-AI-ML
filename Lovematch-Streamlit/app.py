import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model
model = pickle.load(open("best_love_model.pkl", "rb"))

# App title
st.title("💘 AI Love Compatibility Predictor")

st.markdown("### 💑 Will you be their **Perfect Match**?")
st.markdown("Rate yourself honestly on the following attributes and let's see if sparks will fly! ✨")

# Input sliders
looks = st.slider("😎 Looks (1-10)", 1, 10, 5)
humor = st.slider("😂 Sense of Humor (1-10)", 1, 10, 5)
kindness = st.slider("🤗 Kindness (1-10)", 1, 10, 5)
confidence = st.slider("💪 Confidence (1-10)", 1, 10, 5)
intelligence = st.slider("🧠 Intelligence (1-10)", 1, 10, 5)

# Prepare input
user_input = np.array([[looks, humor, kindness, confidence, intelligence]])

# Predict button
if st.button("💘 Predict Love Compatibility"):
    prediction = model.predict(user_input)
    
    if prediction[0] == 1:
        st.success("💖 It's a Match! They're definitely into you!")
        st.balloons()
    else:
        st.error("💔 Oops! Better luck next time — stay awesome!")
