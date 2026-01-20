import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import tempfile

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Pneumonia Detection | Tejas Gholap",
    page_icon="🫁",
    layout="wide"
)

# ================== PREMIUM CSS ==================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
}

.glass {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 18px;
    padding: 25px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

.title {
    font-size: 46px;
    font-weight: 800;
    color: #ffffff;
}

.subtitle {
    font-size: 18px;
    color: #d1d5db;
}

.button > button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 12px;
    padding: 12px 24px;
    font-size: 18px;
    font-weight: 600;
}

.profile a {
    color: #00c6ff;
    text-decoration: none;
    font-weight: 600;
}

.footer {
    text-align: center;
    color: #9ca3af;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ================== HEADER ==================
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.markdown('<div class="title">🫁 Pneumonia Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Chest X-Ray Analysis using YOLO</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ================== MAIN LAYOUT ==================
col1, col2 = st.columns([2.2, 1])

# ================== LEFT PANEL ==================
with col1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📤 Upload Chest X-Ray")

    uploaded = st.file_uploader(
        "Supported formats: JPG, PNG",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_column_width=True)

        if st.button("🚀 Run Detection"):
            with st.spinner("AI is analyzing the X-Ray..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    results = model(tmp.name, conf=0.25)

                result_img = results[0].plot()
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

                st.success("Detection Completed")
                st.image(result_img, use_column_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ================== RIGHT PANEL ==================
with col2:
    st.markdown('<div class="glass profile">', unsafe_allow_html=True)
    st.subheader("👨‍💻 Developer")

    st.markdown("""
**Tejas Gholap**  
📧 tejasgholap961@gmail.com  

🔗 [LinkedIn](https://www.linkedin.com/in/tejas-gholap-bb3417300/)  
💻 [GitHub](https://github.com/tejasgholap45)  
🌐 [Portfolio](https://tejas-gholap-data-analys-2x22p9s.gamma.site/)
""")

    st.markdown("---")
    st.markdown("""
**Tech Stack**
- Python
- YOLO (Ultralytics)
- Computer Vision
- Streamlit
- Deep Learning
""")

    st.markdown('</div>', unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown('<div class="footer">© 2026 | Built by Tejas Gholap</div>', unsafe_allow_html=True)
