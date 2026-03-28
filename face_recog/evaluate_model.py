import os
import cv2
import numpy as np
import pickle
from deepface import DeepFace
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
# SETTINGS
TEST_FOLDER = "Face-recognition/Test_Evaluation"
THRESHOLD = 0.6

# Load face detector
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load DeepFace model
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_path = os.path.join(BASE_DIR, "deepface_model", "ArcFace_model.pkl")

with open(model_path, "rb") as f:
    deepface_data = pickle.load(f)

deepface_data["embeddings"] = np.array(deepface_data["embeddings"])
deepface_data["labels"] = np.array(deepface_data["labels"])

saved_embeddings = deepface_data["embeddings"]

id_to_name = {v: k for k, v in deepface_data["label_to_id"].items()}

true_labels = []
predicted_labels = []

print("\nRunning evaluation...\n")

images = [f for f in os.listdir(TEST_FOLDER) if f.endswith((".jpg",".png",".jpeg"))]

for img_name in tqdm(images):

    true_label = img_name.split("_")[0]

    path = os.path.join(TEST_FOLDER, img_name)
    img = cv2.imread(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = facedetect.detectMultiScale(gray,1.1,5)

    if len(faces) == 0:
        predicted = "unknown"
    else:

        x,y,w,h = faces[0]
        face = img[y:y+h, x:x+w]

        try:

            rep = DeepFace.represent(
                img_path=face,
                model_name=deepface_data["model_name"],
                enforce_detection=False
            )

            emb = np.array(rep[0]["embedding"])
            emb = emb / np.linalg.norm(emb)

            sims = [1 - cosine(emb, e) for e in saved_embeddings]

            best_idx = int(np.argmax(sims))
            best_sim = sims[best_idx]

            best_label = deepface_data["labels"][best_idx]
            name = id_to_name.get(best_label,"unknown")

            if best_sim < THRESHOLD:
                predicted = "unknown"
            else:
                predicted = name

        except:
            predicted = "unknown"

    true_labels.append(true_label)
    predicted_labels.append(predicted)

# METRICS
accuracy = accuracy_score(true_labels,predicted_labels)
precision = precision_score(true_labels,predicted_labels,average="weighted",zero_division=0)
recall = recall_score(true_labels,predicted_labels,average="weighted",zero_division=0)
f1 = f1_score(true_labels,predicted_labels,average="weighted",zero_division=0)

print("\n===== MODEL EVALUATION =====")

print(f"Accuracy  : {accuracy:.3f}")
print(f"Precision : {precision:.3f}")
print(f"Recall    : {recall:.3f}")
print(f"F1 Score  : {f1:.3f}")

# CONFUSION MATRIX
labels = list(set(true_labels + predicted_labels))

cm = confusion_matrix(true_labels,predicted_labels,labels=labels)

plt.figure(figsize=(8,6))
sns.heatmap(cm,annot=True,fmt="d",xticklabels=labels,yticklabels=labels,cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.show()
