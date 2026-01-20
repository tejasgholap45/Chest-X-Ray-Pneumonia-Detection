import gradio as gr
from ultralytics import YOLO
from PIL import Image
import tempfile

# Load YOLO model
model = YOLO("best.pt")

def detect_pneumonia(image):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        results = model(tmp.name, conf=0.25)

    output_image = results[0].plot(pil=True)
    return output_image

# Gradio UI
interface = gr.Interface(
    fn=detect_pneumonia,
    inputs=gr.Image(type="pil", label="Upload Chest X-Ray"),
    outputs=gr.Image(type="pil", label="Prediction Output"),
    title="🫁 Chest X-Ray Pneumonia Detection",
    description="""
AI-powered Pneumonia Detection using YOLO

👨‍💻 **Developer:** Tejas Gholap  
📧 Email: tejasgholap961@gmail.com  
🔗 LinkedIn: https://www.linkedin.com/in/tejas-gholap-bb3417300/  
💻 GitHub: https://github.com/tejasgholap45  
🌐 Portfolio: https://tejas-gholap-data-analys-2x22p9s.gamma.site/
"""
)

if __name__ == "__main__":
    interface.launch()
