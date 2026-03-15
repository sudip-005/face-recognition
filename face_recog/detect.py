import cv2
import numpy as np
import os
import json
import sys
import pickle
# Reduce TensorFlow/DeepFace startup noise
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from deepface import DeepFace
from scipy.spatial.distance import cosine
from sklearn.preprocessing import Normalizer

# Load the Haar Cascade for face detection
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if facedetect.empty():
    raise ValueError("ERROR: Error loading Haar Cascade. Check file path.")

print('DEBUG: detect.py module loaded')


def create_recognizer():
    """Attempt to create an LBPH face recognizer. If opencv-contrib-python
    is not available, returns None and prints actionable instructions."""
    try:
        # Preferred: opencv-contrib-python exposes cv2.face
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        return recognizer
    except Exception:
        # Inform the user how to install the required package
        print("ERROR: cv2.face module not found. Please install opencv-contrib-python:")
        print("   pip install opencv-contrib-python")
        return None


def get_user_info_from_dataset(user_id):
    """
    Get user information from the dataset folder structure
    """
    try:
        # Look for user folder with matching ID
        dataset_path = "dataset"
        if not os.path.exists(dataset_path):
            return None
        
        # Search for folder starting with user_id
        for folder in os.listdir(dataset_path):
            if folder.startswith(str(user_id) + "_"):
                # Format: ID_Name
                parts = folder.split('_', 1)
                if len(parts) == 2:
                    name = parts[1]
                    return (user_id, name, "Unknown")
        
        # Check metadata.json if exists (folder named exactly by user_id)
        metadata_file = os.path.join(dataset_path, str(user_id), "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                data = json.load(f)
                return (data.get('id'), data.get('name'), data.get('age', 'Unknown'))
        
        return None
    except Exception as e:
        print(f"WARNING: Error reading user info: {e}")
        return None


def get_user_info_from_filename(user_id):
    """
    Alternative: Get user info from image filenames in dataset
    """
    try:
        dataset_path = "dataset"
        if not os.path.exists(dataset_path):
            return None
        
        # Search for any image with this user_id
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.startswith(f"{user_id}_") and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Extract name from filename (format: ID_Name_number.jpg)
                    parts = file.split('_')
                    if len(parts) >= 2:
                        name = parts[1]
                        return (user_id, name, "Unknown")
    except:
        pass
    return None


def main():
    # Start video capture
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: Could not open camera. Ensure a webcam is connected and accessible.")
        sys.exit(1)

    # Set camera resolution for better performance
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Attempt to load DeepFace pickle model first
    # Resolve paths relative to project root (one level above this script)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    deepface_pkl = os.path.join(base_dir, "deepface_model", "ArcFace_model.pkl")
    use_deepface = False
    deepface_data = None
    if os.path.exists(deepface_pkl):
        try:
            with open(deepface_pkl, 'rb') as f:
                deepface_data = pickle.load(f)
            # Ensure embeddings are numpy array
            deepface_data['embeddings'] = np.array(deepface_data.get('embeddings', []))
            deepface_data['labels'] = np.array(deepface_data.get('labels', []))
            use_deepface = True
            print(f"OK: Loaded DeepFace model from {deepface_pkl}")
        except Exception as e:
            print(f"WARNING: Failed to load {deepface_pkl}: {e}")

    # If DeepFace model not available, fallback to LBPH recognizer
    recognizer = None
    if not use_deepface:
        recognizer = create_recognizer()
        if recognizer is None:
            sys.exit(1)

        model_path = os.path.join(base_dir, "recognizer", "trainingdata.yml")
        try:
            if not os.path.exists(model_path):
                print(f"ERROR: Model file not found at {model_path}")
                print("   Train a model first or place trainingdata.yml in the recognizer folder.")
                sys.exit(1)

            recognizer.read(model_path)
            print(f"OK: LBPH model loaded successfully from {model_path}")
        except Exception as e:
            print(f"ERROR: Error loading training data: {e}")
            sys.exit(1)

    # Create a cache for user profiles to avoid repeated disk access
    profile_cache = {}

    print("\n" + "="*60)
    print("INFO: FACE RECOGNITION SYSTEM - READY")
    print("="*60)
    print("Camera started. Press 'q' to quit")
    print("Detected faces will be identified")
    print("="*60)

    while True:
        ret, img = cam.read()
        if not ret:
            print("ERROR: Failed to capture image")
            break

        # Flip horizontally for mirror effect
        img = cv2.flip(img, 1)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(100, 100)
        )

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add "Face" label
            cv2.putText(img, "Face Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Predict face
            try:
                if use_deepface and deepface_data is not None:
                    # Use DeepFace embeddings for matching
                    face_img = img[y:y + h, x:x + w]
                    rep = DeepFace.represent(img_path=face_img, model_name=deepface_data.get('model_name', 'ArcFace'),
                                             enforce_detection=False, detector_backend='opencv', align=True)
                    if rep and len(rep) > 0:
                        emb = np.array(rep[0]['embedding'])
                        l2 = np.linalg.norm(emb)
                        if l2 > 0:
                            emb = emb / l2

                        saved_embs = deepface_data['embeddings']
                        if saved_embs.shape[0] == 0:
                            raise ValueError('No embeddings in DeepFace model')

                        sims = [1 - cosine(emb, e) for e in saved_embs]
                        best_idx = int(np.argmax(sims))
                        best_sim = sims[best_idx]
                        best_label = int(deepface_data['labels'][best_idx]) if 'labels' in deepface_data else None
                        id_to_name = {v: k for k, v in deepface_data.get('label_to_id', {}).items()}
                        name = id_to_name.get(best_label, f'ID_{best_label}') if best_label is not None else 'Unknown'

                        if best_sim > 0.85:
                            quality = 'Excellent'
                            color = (0, 255, 0)
                        elif best_sim > 0.7:
                            quality = 'Good'
                            color = (0, 255, 255)
                        elif best_sim > 0.6:
                            quality = 'Fair'
                            color = (0, 165, 255)
                        else:
                            quality = 'Poor'
                            color = (0, 0, 255)

                        if best_sim > 0.6:
                            cv2.putText(img, f"ID: {best_label}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(img, f"Name: {name}", (x, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            cv2.putText(img, f"Similarity: {best_sim:.3f}", (x, y + h + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.putText(img, f"Quality: {quality}", (x, y + h + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            cv2.circle(img, (x + w - 20, y + 20), 10, color, -1)
                        else:
                            cv2.putText(img, "Unknown", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            cv2.putText(img, f"Similarity: {best_sim:.3f}", (x, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            cv2.circle(img, (x + w - 20, y + 20), 10, (0, 0, 255), -1)
                    else:
                        cv2.putText(img, "No face embedding", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # LBPH fallback
                    user_id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                    # Get user profile (with caching)
                    if user_id not in profile_cache:
                        profile = get_user_info_from_dataset(user_id)
                        if profile is None:
                            profile = get_user_info_from_filename(user_id)
                        profile_cache[user_id] = profile
                    else:
                        profile = profile_cache[user_id]

                    if confidence < 50:
                        quality = "Excellent"
                        color = (0, 255, 0)
                    elif confidence < 70:
                        quality = "Good"
                        color = (0, 255, 255)
                    elif confidence < 85:
                        quality = "Fair"
                        color = (0, 165, 255)
                    else:
                        quality = "Poor"
                        color = (0, 0, 255)

                    if profile is not None:
                        name = profile[1]
                        cv2.putText(img, f"ID: {user_id}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(img, f"Name: {name}", (x, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(img, f"Confidence: {confidence:.1f}", (x, y + h + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.putText(img, f"Quality: {quality}", (x, y + h + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.circle(img, (x + w - 20, y + 20), 10, color, -1)
                    else:
                        cv2.putText(img, f"ID: {user_id} (Unknown)", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.putText(img, f"Confidence: {confidence:.1f}", (x, y + h + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.circle(img, (x + w - 20, y + 20), 10, (0, 0, 255), -1)
            except Exception as e:
                print(f"WARNING: Prediction error: {e}")
                cv2.putText(img, "Recognition Error", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Add status bar at the top
        cv2.rectangle(img, (0, 0), (640, 40), (0, 0, 0), -1)
        cv2.putText(img, "Face Recognition System", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, "Press 'q' to quit", (480, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("FACE RECOGNITION SYSTEM", img)

        if cv2.waitKey(1) == ord('q'):
            print("\nExiting...")
            break

    # Release camera and close windows
    cam.release()
    cv2.destroyAllWindows()

    print("\n" + "="*60)
    print("OK: Recognition session ended")
    print("="*60)


if __name__ == "__main__":
    main()
