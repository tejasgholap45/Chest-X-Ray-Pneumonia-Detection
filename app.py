import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Pneumonia Detection | Tejas Gholap",
    page_icon="🫁",
    layout="wide"
)

st.title("🫁 Chest X-Ray Pneumonia Detection")
st.caption("AI-powered Chest X-Ray Analysis using YOLO")
st.divider()

# ================== LOAD MODEL ==================
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ================== LAYOUT ==================
left, right = st.columns(2)

# ================== INPUT ==================
with left:
    st.subheader("📥 Input X-Ray Image")
    uploaded = st.file_uploader(
        "Upload Chest X-Ray (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        input_image = Image.open(uploaded).convert("RGB")
        st.image(input_image, use_column_width=True)

# ================== OUTPUT ==================
with right:
    st.subheader("📤 Model Prediction")

    if uploaded:
        if st.button("Run Detection"):
            with st.spinner("Analyzing X-Ray..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    input_image.save(tmp.name)
                    results = model(tmp.name, conf=0.25)

                # ✅ PIL output — NO cv2, NO libGL
                output_image = results[0].plot(pil=True)

                st.success("Detection Completed ✅")
                st.image(output_image, use_column_width=True)
    else:
        st.info("Upload image to see prediction")

st.divider()
st.markdown("""
**Developer:** Tejas Gholap  
📧 tejasgholap961@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/tejas-gholap-bb3417300/  
💻 GitHub: https://github.com/tejasgholap45  
🌐 Portfolio: https://tejas-gholap-data-analys-2x22p9s.gamma.site/
""")
