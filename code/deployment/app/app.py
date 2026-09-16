import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def call_api(image_bytes, filename, content_type):
    files = {"file": (filename, image_bytes, content_type or "image/png")}
    response = requests.post(f"{API_URL}/predict", files=files, timeout=60)
    response.raise_for_status()
    return response.json()


st.title("MNIST digit recognition")
st.write("Upload a photo of a handwritten digit. The model API will predict the digit.")

uploaded = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    st.image(uploaded)

if st.button("Predict", disabled=uploaded is None):
    try:
        result = call_api(uploaded.getvalue(), uploaded.name, uploaded.type)
        st.success(f"Predicted digit: {result['digit']}")
        st.write(f"Confidence: {result['confidence']:.2%}")
        probs = {str(k): v for k, v in sorted(result["probabilities"].items())}
        st.bar_chart(probs)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
