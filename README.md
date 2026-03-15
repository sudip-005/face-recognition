# 🎭 Face Recognition System (Face_Recog)

A lightweight **Face Recognition Toolkit** built with **DeepFace (ArcFace)** and **OpenCV**, designed for real-time recognition, dataset creation, and model evaluation.

This project allows you to:

* 📸 Collect face datasets
* 🧠 Train embeddings using **ArcFace**
* 🎥 Perform **real-time face recognition**
* 🧪 Run **blind / batch testing on images**
* 🔁 Use **LBPH fallback recognition** if DeepFace embeddings are unavailable

---

# 🚀 Features

✨ **Dataset Creation**

* Capture face images from webcam
* Automatic folder organization per user

🧠 **Deep Learning Recognition**

* Uses **ArcFace embeddings via DeepFace**
* High accuracy facial feature matching

⚡ **Fallback Recognition**

* Uses **OpenCV LBPH recognizer** when DeepFace model is unavailable

📷 **Real-Time Detection**

* Live webcam face recognition

🧪 **Blind Testing**

* Predict faces from single images or batches

📊 **Evaluation Ready**

* Designed to integrate with model evaluation scripts (accuracy, confusion matrix)

---

# 📂 Project Structure

```
Face_Recog
│
├── dataset/                   # Training images (per user)
│   ├── 001_Name/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── metadata.json
│
├── deepface_model/            # Saved DeepFace embeddings
│   ├── ArcFace_model.pkl
│   └── ArcFace_metadata.json
│
├── recognizer/                # OpenCV LBPH model
│   └── trainingdata.yml
│
├── Face-recognition/          # Core scripts
│   ├── dataset_creator.py
│   ├── trainer.py
│   ├── detect.py
│   ├── test_detect.py
│   ├── evaluate_model.py
│   └── haarcascade_frontalface_default.xml
│
├── .gitignore
└── README.md
```

---

# 🛠️ Requirements

Recommended environment:

* **Python 3.10+**
* Windows / Linux / macOS

Required Python packages:

```
deepface
opencv-python
opencv-contrib-python
numpy
scipy
scikit-learn
matplotlib
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/your-username/face-recognition-system.git
cd face-recognition-system
```

Create a virtual environment:

```
python -m venv .venv
```

Activate environment:

### Windows

```
.venv\Scripts\Activate.ps1
```

### Linux / Mac

```
source .venv/bin/activate
```

Install dependencies:

```
pip install --upgrade pip
pip install deepface opencv-python opencv-contrib-python numpy scipy scikit-learn matplotlib
```

⚠ **Note**

DeepFace may install **TensorFlow**, which can take some time.

---

# 📸 Step 1 — Create Dataset

Capture images using webcam:

```
python Face-recognition/dataset_creator.py
```

This will create user folders like:

```
dataset/
  ├── 001_John/
  ├── 002_Alice/
```

---

# 🧠 Step 2 — Train Face Embeddings

Train ArcFace embeddings:

```
python Face-recognition/trainer.py
```

Output:

```
deepface_model/
  ArcFace_model.pkl
  ArcFace_metadata.json
```

---

# 🎥 Step 3 — Real-Time Face Recognition

Run webcam recognition:

```
python Face-recognition/detect.py
```

Features:

* Face detection
* Identity prediction
* Confidence score
* Recognition quality indicator

---

# 🧪 Step 4 — Blind Image Testing

Run prediction on a folder of images:

```
python Face-recognition/test_detect.py --input Face-recognition/Test_Images --output results.json
```

Single image prediction:

```
python Face-recognition/test_detect.py Face-recognition/Test_Images/photo.jpg
```

Output example:

```
Prediction Result
------------------
ID: 3
Name: Kajol
Similarity: 0.87
```

---

# 🧠 Recognition Logic

The system uses:

### Primary Model

**DeepFace ArcFace embeddings**

Similarity score is computed using **cosine similarity**.

### Fallback Model

If DeepFace model is missing:

```
recognizer/trainingdata.yml
```

OpenCV **LBPH face recognition** is used.

---

# 🧹 Cleanup (Safe to Remove)

You can safely delete:

```
__pycache__/
*.pyc
Face-recognition/output_log.txt
```

IDE folders such as:

```
.idea/
.vscode/
```

---

# ⚠ Troubleshooting

### TensorFlow logs appearing

Run before executing scripts:

```
$env:TF_CPP_MIN_LOG_LEVEL='3'
$env:TF_ENABLE_ONEDNN_OPTS='0'
```

---

### LBPH recognizer missing

Install OpenCV contrib:

```
pip install opencv-contrib-python
```

---

### Webcam not detected

Check Windows settings:

```
Settings → Privacy → Camera → Enable access
```

---

# 📈 Future Improvements

Potential enhancements:

* 📊 Model evaluation dashboard
* 🌐 Web interface (Flask / Streamlit)
* 📉 ROC curves and confusion matrix
* 📱 Face recognition API
* 🧠 Multiple embedding models (ArcFace / Facenet)

---

# 👨‍💻 Author

**Sudip Manna**

Machine Learning & Computer Vision enthusiast.

---

# ⭐ Support

If you found this project useful:

⭐ Star the repository
🍴 Fork the project
🧠 Contribute improvements

---

# 📜 License

This project is open-source and available under the **MIT License**.
