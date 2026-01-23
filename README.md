# 🫁 Chest X-Ray Pneumonia Detection Web App

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLO](https://img.shields.io/badge/Model-YOLOv8-orange)
![Framework](https://img.shields.io/badge/Web-Gradio-green)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Deployment](https://img.shields.io/badge/Deployed-HuggingFace-blue)

A production-ready deep learning web application for detecting pneumonia from chest X-ray images using YOLO and Gradio.  
The model is deployed on Hugging Face Spaces for real-time inference.

---

## 🚀 Live Demo  
👉 https://huggingface.co/spaces/Tejas04580/chest-xray-pneumonia-detection  

---

## 📌 Project Overview  

Pneumonia is a serious respiratory infection that can be diagnosed using chest X-ray imaging.  
This project leverages a **YOLO-based object detection model** to automatically detect pneumonia regions in chest X-rays and provides a **web-based inference platform** for real-time usage.

The goal of this project is to demonstrate **end-to-end AI model development and deployment**, including model training, inference, and cloud deployment.

---

## 🧠 Tech Stack  

- **Language:** Python  
- **Deep Learning:** PyTorch  
- **Model:** YOLO (Ultralytics)  
- **Web UI:** Gradio  
- **Image Processing:** Pillow, OpenCV  
- **Deployment:** Hugging Face Spaces  
- **Version Control:** Git & GitHub  

---

## ⚙️ Features  

- Upload chest X-ray images (PNG/JPG)  
- Real-time pneumonia detection  
- Bounding box visualization  
- Interactive web-based UI  
- Cloud deployment with public access  
- Scalable and reproducible ML pipeline  

---

📂 Project Structure
---

Chest-X-Ray-Pneumonia-Detection/

│

├── app.py                # Gradio web application

├── best.pt               # Trained YOLO model weights

├── requirements.txt      # Dependencies

├── README.md              # Documentation

└── LICENSE                # Project licens


▶️ Run Locally
---

1️⃣ Clone Repository
git clone https://github.com/tejasgholap45/Chest-X-Ray-Pneumonia-Detection.git
cd Chest-X-Ray-Pneumonia-Detection

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Application
python app.py

---

📊 Model Details
---

Model Architecture: YOLO Object Detection

Task: Pneumonia detection in chest X-rays

Dataset: Public Chest X-Ray dataset

Metrics: Precision, Recall, mAP

Framework: PyTorch

---

🏗️ System Architecture
---
User Image Upload → YOLO Model Inference → Bounding Box Detection → Web UI Output

---

⚠️ Disclaimer
---
This project is intended for educational and research purposes only.
It is not a medical diagnostic tool and should not be used for clinical decisions.

---

📌 Future Enhancements
---
Pneumonia vs Normal classification with confidence score

Downloadable prediction reports

Analytics dashboard

Cloud deployment on AWS/GCP

Model explainability (Grad-CAM)

---

👨‍💻 Developer
---

Tejas Gholap

📧 Email: tejasgholap961@gmail.com

🔗 LinkedIn: https://www.linkedin.com/in/tejas-gholap-bb3417300/

💻 GitHub: https://github.com/tejasgholap45

🌐 Portfolio: https://tejas-gholap-data-analys-2x22p9s.gamma.site/

---

📜 License
---

This project is licensed under the MIT License – see the LICENSE file for details.

---

⭐ Acknowledgements
---
Ultralytics YOLO

Public Chest X-Ray Dataset

Hugging Face Spaces
