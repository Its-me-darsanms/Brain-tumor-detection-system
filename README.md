**Brain Tumor Detection System 🧠**

A lightweight, Streamlit-based app that detects brain tumors in MRI scans using a YOLOv8 model.
Upload an MRI image and the app will run YOLO detection, draw bounding boxes on regions of interest, and report confidence scores.

🚀 Features

Fast inference with YOLOv8 (.pt model file)

Simple web UI built with Streamlit

Shows original MRI and AI-annotated image side-by-side

Displays number of detected regions and confidence for each detection

Clear disclaimer: research tool, not a medical diagnosis

🔧 Tech Stack

Python 3.10+

Streamlit

Ultralytics YOLO (YOLOv8)

OpenCV (cv2)

Pillow (PIL)

NumPy

PyTorch (CPU or GPU backend depending on your machine)

📁 Repo structure
Brain-tumor-detection-system/
│── app.py                   # Streamlit app (UI + inference)
│── brain_tumor_model.pt     # Trained YOLOv8 model (weights)
│── requirements.txt         # Python dependencies
│── README.md                # This file

⚙️ Installation & Quick Start

Recommended: create a virtual environment (venv / conda) before installing dependencies.

Clone the repository

git clone https://github.com/your-username/Brain-tumor-detection-system.git
cd Brain-tumor-detection-system


Create and activate a virtual environment (optional but recommended)

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate


Install dependencies

pip install -r requirements.txt


Run the Streamlit app

streamlit run app.py


The app will open in your browser at http://localhost:8501 (or the link shown in the terminal).

🧠 About the Model

Format: YOLOv8 .pt weights (pretrained/fine-tuned for brain tumor detection).

Input preprocessing: the ultralytics.YOLO wrapper handles resizing and normalization for the model (default YOLOv8 behavior).

Inference: results = model(image) (where image is a PIL image loaded from upload).

Detection output: results[0].boxes — each box contains coordinates, confidence score, and class id (if multi-class).

Model info shown in the app sidebar:

Type: YOLOv8

Input size: 640×640 (typical YOLO input used during training/inference)

Task: Brain tumor region detection

Replace brain_tumor_model.pt with your trained model. Ensure the model was exported/trained with a YOLOv8-compatible config.

🖼️ How to use the app

Open the Streamlit page in your browser.

Upload an MRI image (.jpg, .jpeg, .png).

Click Detect Tumors.

The app will show:

Original MRI (left column)

Annotated image with bounding boxes (right column)

A message ⛔ TUMOR DETECTED with the number of regions and per-region confidence if detections exist; otherwise ✅ NO TUMOR DETECTED.

The app uses results[0].plot() to render bounding boxes on the image.

🧾 Example output (what you’ll see)

🚨 TUMOR DETECTED - 2 region(s) found

Region 1: 87.2% confidence

Region 2: 65.4% confidence

Or: ✅ NO TUMOR DETECTED if len(results[0].boxes) == 0.

✅ Notes & Tips

GPU recommendation: If you have an NVIDIA GPU and want faster inference, install CUDA-enabled PyTorch. If you run on CPU, inference will be slower but still functional.

Model compatibility: The app expects a YOLOv8 .pt file compatible with ultralytics.YOLO. If you trained with a different pipeline, re-export to YOLOv8 format.

Input size: YOLOv8 models are typically trained with 640×640 input. If your model expects a different size, ensure consistency (or retrain/export appropriately).

Image formats: Use clear MRI scans (non-blurry, good contrast). Preprocessing like CLAHE/contrast adjustment can help model performance but is not applied by default.

⚠️ Disclaimer

This tool is built for research, learning, and demonstration purposes only. It is not a substitute for professional medical diagnostics. Always consult certified healthcare professionals for diagnosis and treatment.

🛠️ Troubleshooting

App crashes with model load error: Confirm brain_tumor_model.pt is present in the repo root and is a valid YOLOv8 model. Check Python & ultralytics versions.

ultralytics import errors: Make sure you have installed the ultralytics package from PyPI: pip install ultralytics.

Slow performance: Use a machine with GPU and install corresponding PyTorch with CUDA support.

Streamlit caching issues: If the model updates and the cached resource causes old model to be used, restart the Streamlit app (Ctrl+C in terminal and re-run streamlit run app.py).

📦 requirements.txt (example)

Make sure your requirements.txt contains versions used during development. Example:

streamlit>=1.20
ultralytics>=8.0
torch>=1.12
opencv-python
pillow
numpy


Adjust versions as needed for your environment (especially PyTorch + CUDA).

🔁 Future Improvements

Add Grad-CAM or heatmap overlays to show model attention regions.

Add batch image upload & bulk inference.

Add confidence threshold slider to filter weak detections.

Add model info (mAP, precision/recall) and training logs.

Deploy as a cloud app (Heroku / Railway / Streamlit Community Cloud / AWS / GCP).

👨‍💻 Author

Darsan M S — IT Student SRM, Ramapuram
Check more projects on my GitHub: https://github.com/its-me-darsanms/
