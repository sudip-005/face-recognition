# Face Recognition (Face_Recog)

Lightweight face-recognition toolkit using DeepFace (ArcFace) with an LBPH fallback.

## Overview

- Collect face images with `Face-recognition/dataset_creator.py`.
- Train DeepFace embeddings with `Face-recognition/trainer.py` (saves into `deepface_model/`).
- Real-time recognition with `Face-recognition/detect.py`.
- Batch / blind predictions (no GUI) with `Face-recognition/test_detect.py`.

## Repository layout

- dataset/ — per-user folders (e.g. `001_Name/`) with `metadata.json`.
- deepface_model/ — saved DeepFace pickle and metadata.
- Face-recognition/ — main scripts: `dataset_creator.py`, `trainer.py`, `detect.py`, `test_detect.py`, utilities.
- recognizer/ — LBPH `trainingdata.yml` (OpenCV).

See these files: [Face-recognition/dataset_creator.py](Face-recognition/dataset_creator.py), [Face-recognition/trainer.py](Face-recognition/trainer.py), [Face-recognition/detect.py](Face-recognition/detect.py), [Face-recognition/test_detect.py](Face-recognition/test_detect.py).

## Requirements

- Python 3.10+ recommended
- pip packages: `deepface`, `opencv-python`, `opencv-contrib-python` (for LBPH), `numpy`, `scipy`, `scikit-learn`, `matplotlib` (optional)

Example install (Windows PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install deepface opencv-python opencv-contrib-python numpy scipy scikit-learn matplotlib
```

Note: DeepFace may pull a backend such as TensorFlow; expect additional dependencies and longer install times.

## Quick start

1. Collect dataset (interactive):

```powershell
python Face-recognition/dataset_creator.py
```

2. Train DeepFace embeddings:

```powershell
python Face-recognition/trainer.py
```

3. Run real-time recognition (webcam):

```powershell
python Face-recognition/detect.py
```

4. Run blind/batch predictions on images (no GUI):

```powershell
python Face-recognition/test_detect.py --input Face-recognition/Test_Images --output results.json
```

Or single image:

```powershell
python Face-recognition/test_detect.py Face-recognition/Test_Images/photo.jpg
```

## Models & files

- If `deepface_model/{Model}_model.pkl` exists, the scripts will prefer DeepFace embeddings.
- If not available, `recognizer/trainingdata.yml` (LBPH) is used as a fallback — requires `opencv-contrib-python`.

## Cleanup / unnecessary files

- It's safe to remove generated caches and non-source files: `__pycache__/`, `*.pyc`, `Face-recognition/output_log.txt`.
- Do NOT remove `deepface_model/` or `recognizer/trainingdata.yml` unless you have backups.
- IDE folders (e.g. `.idea/`) are optional and can be removed if you don't use that IDE.

## Troubleshooting

- If DeepFace triggers heavy TensorFlow logs, set these env vars before running (PowerShell):

```powershell
$env:TF_CPP_MIN_LOG_LEVEL='3'; $env:TF_ENABLE_ONEDNN_OPTS='0'
```

- If LBPH is missing, install `opencv-contrib-python`.
- For camera permission issues on Windows, ensure the app has access to the webcam.

## Next steps

- Add `requirements.txt` if you want a reproducible environment.
- Add CSV export or a small web UI for batch review of predictions.

---
Created for the Face_Recog workspace. If you want a more detailed README (installation guide, examples, or CI), tell me which sections to expand.
