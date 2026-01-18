import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Pneumonia Detection | Tejas Gholap",
    page_icon="🫁",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f8f9fa;
}
.profile-card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
}
.title-text {
    font-size: 40px;
    font-weight: 700;
    color: #0d6efd;
}
.subtitle-text {
    font-size: 18px;
    color: #555;
}
.footer {
    text-align: center;
    padding: 10px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title-text'>🫁 Chest X-Ray Pneumonia Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Deep Learning based YOLO Object Detection App</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([2, 1])

# ---------------- IMAGE UPLOAD ----------------
with col1:
    st.subheader("📤 Upload Chest X-Ray Image")
    uploaded_file = st.file_uploader(
        "Upload X-Ray Image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)

        if st.button("🔍 Detect Pneumonia"):
            with st.spinner("Running model inference..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    results = model(tmp.name, conf=0.25)

                res_img = results[0].plot()
                res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)

                st.success("Detection Completed ✅")
                st.image(res_img, caption="Detection Result", use_column_width=True)

# ---------------- PROFILE SECTION ----------------
with col2:
    st.markdown("<div class='profile-card'>", unsafe_allow_html=True)
    st.subheader("👨‍💻 Developer Profile")

    st.markdown("""
**Name:** Tejas Gholap  
📧 **Email:** tejasgholap961@gmail.com  

🔗 **LinkedIn:**  
https://www.linkedin.com/in/tejas-gholap-bb3417300/

💻 **GitHub:**  
https://github.com/tejasgholap45  

🌐 **Portfolio:**  
https://tejas-gholap-data-analys-2x22p9s.gamma.site/
""")

    st.markdown("---")
    st.markdown("""
**Skills Used**
- Python
- YOLO (Ultralytics)
- Computer Vision
- Deep Learning
- Streamlit
""")

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<div class='footer'>© 2026 | Pneumonia Detection App by Tejas Gholap</div>",
    unsafe_allow_html=True
)
