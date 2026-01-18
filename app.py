import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import os

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

button {
    background: linear-gradient(90deg, #00c6ff, #0072ff) !important;
    color: white !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
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
st.markdown('<div class="title">🫁 Chest X-Ray Pneumonia Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Pneumonia Detection using YOLO</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ================== MAIN LAYOUT ==================
left, right, profile = st.columns([2, 2, 1])

# ================== INPUT ==================
with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📥 Input X-Ray Image")

    uploaded = st.file_uploader(
        "Upload Chest X-Ray (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        input_image = Image.open(uploaded).convert("RGB")
        st.image(input_image, use_column_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ================== OUTPUT ==================
with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📤 Model Prediction")

    if uploaded:
        if st.button("🚀 Run Detection"):
            with st.spinner("Analyzing X-Ray using AI..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    input_image.save(tmp.name)
                    results = model(tmp.name, conf=0.25)

                output_img = results[0].plot()
                output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

                st.success("Detection Completed ✅")
                st.image(output_img, use_column_width=True)
    else:
        st.info("Upload an image to see prediction")

    st.markdown('</div>', unsafe_allow_html=True)

# ================== PROFILE ==================
with profile:
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
- Deep Learning  
- Streamlit  
""")

    st.markdown('</div>', unsafe_allow_html=True)

# ================== FOOTER ==================
st.markdown('<div class="footer">© 2026 | Built by Tejas Gholap</div>', unsafe_allow_html=True)
