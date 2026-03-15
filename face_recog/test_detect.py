import os
import sys
import argparse
import json
import pickle
import cv2
import numpy as np
from scipy.spatial.distance import cosine


# Reduce TensorFlow logs when DeepFace is imported
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')


# Load Haar Cascade
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if facedetect.empty():
    raise ValueError("ERROR: Could not load Haar Cascade")


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_deepface_model(model_name="ArcFace"):
    model_path = os.path.join(BASE_DIR, "deepface_model", f"{model_name}_model.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    data["embeddings"] = np.array(data.get("embeddings", []))
    data["labels"] = np.array(data.get("labels", []))
    print(f"OK: DeepFace model loaded from {model_path}")
    return data


def load_lbph_model():
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except Exception:
        print("ERROR: opencv-contrib-python (cv2.face) is required for LBPH fallback")
        return None
    model_path = os.path.join(BASE_DIR, "recognizer", "trainingdata.yml")
    if not os.path.exists(model_path):
        print("WARN: LBPH model file not found: recognizer/trainingdata.yml")
        return None
    recognizer.read(model_path)
    print(f"OK: LBPH model loaded from {model_path}")
    return recognizer


def predict_deepface(face_img, deepface_data, DeepFace_lib):
    # DeepFace_lib is the imported DeepFace module (or None)
    if DeepFace_lib is None:
        return None
    try:
        rep = DeepFace_lib.represent(
            img_path=face_img,
            model_name=deepface_data.get("model_name", "ArcFace"),
            enforce_detection=False,
            detector_backend="opencv",
            align=True,
        )
    except Exception as e:
        print(f"WARN: DeepFace.represent error: {e}")
        return None
    if not rep:
        return None
    emb = np.array(rep[0]["embedding"])
    l2 = np.linalg.norm(emb)
    if l2 > 0:
        emb = emb / l2
    saved_embs = deepface_data.get("embeddings", np.array([]))
    if saved_embs.size == 0:
        return None
    sims = [1 - cosine(emb, e) for e in saved_embs]
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    best_label = int(deepface_data["labels"][best_idx])
    id_to_name = {v: k for k, v in deepface_data.get("label_to_id", {}).items()}
    name = id_to_name.get(best_label, f"ID_{best_label}")
    return best_label, name, best_sim


def predict_lbph(face_img, recognizer):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    try:
        user_id, confidence = recognizer.predict(gray)
        return user_id, float(confidence)
    except Exception as e:
        print(f"WARN: LBPH predict error: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Blind prediction for a single image")
    parser.add_argument('path', nargs='?', help='Image path (positional) or use --input')
    parser.add_argument('--input', '-i', help='Image path or directory')
    parser.add_argument('--model', '-m', default='ArcFace', help='DeepFace model name (default ArcFace)')
    args = parser.parse_args()

    input_path = args.input or args.path
    if not input_path:
        print("Usage: python test_detect.py image.jpg")
        sys.exit(1)
    if not os.path.exists(input_path):
        print("ERROR: Image not found:", input_path)
        sys.exit(1)

    # read image
    img = cv2.imread(input_path)
    if img is None:
        print("ERROR: Could not read image")
        sys.exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        print("No face detected")
        sys.exit(0)

    # try to load deepface model (pickle) and import DeepFace lazily
    deepface_data = load_deepface_model(args.model)
    DeepFace_lib = None
    if deepface_data is not None:
        try:
            from deepface import DeepFace as _DeepFace
            DeepFace_lib = _DeepFace
        except Exception as e:
            print(f"WARN: could not import DeepFace library: {e}")
            DeepFace_lib = None

    recognizer = None
    if deepface_data is None or DeepFace_lib is None:
        recognizer = load_lbph_model()

    # process each detected face
    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        label = None
        score = None
        method = None

        if deepface_data is not None and DeepFace_lib is not None:
            res = predict_deepface(face, deepface_data, DeepFace_lib)
            if res is None:
                print("Could not extract embedding or match")
            else:
                user_id, name, similarity = res
                method = 'deepface'
                score = similarity
                label = name if similarity >= 0.6 else 'UNKNOWN'
                print("\nPrediction Result")
                print("------------------")
                if similarity >= 0.6:
                    print("ID:", user_id)
                    print("Name:", name)
                    print("Similarity:", round(similarity, 3))
                else:
                    print("Prediction: UNKNOWN PERSON")
                    print("Best Similarity:", round(similarity, 3))
        elif recognizer is not None:
            user_id, confidence = predict_lbph(face, recognizer)
            method = 'lbph'
            score = confidence
            label = str(user_id) if user_id is not None else 'UNKNOWN'
            print("\nPrediction Result")
            print("------------------")
            if user_id is not None:
                print("ID:", user_id)
                print("Confidence:", round(confidence, 2))
            else:
                print("LBPH prediction failed")
        else:
            print("No recognizer available")

        # annotate image
        display_label = label if label is not None else 'UNKNOWN'
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, display_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # show result image
    cv2.imshow("Blind Test Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()