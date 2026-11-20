# Combined Face Identification + Facial Expression Recognition

This educational project combines **two different ML tasks**:
- Identity recognition: matches faces you personally register using OpenCV LBPH.
- Expression classification: predicts one of FER-2013's seven dataset labels using a CNN.

**FER-2013 does not identify people's names.** You must collect consenting identity samples separately.

## Requirements
Windows 10/11, Python 3.11 64-bit recommended, webcam, internet for packages/dataset. CPU training can take a long time.

## Install (VS Code terminal / CMD)
```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python check_installation.py
```
Select `.venv` in VS Code: Ctrl+Shift+P -> Python: Select Interpreter.
Install only one OpenCV wheel in this venv; this project needs `opencv-contrib-python` for `cv2.face`.

## FER-2013 dataset setup
Download the dataset from https://www.kaggle.com/datasets/msambare/fer2013 (Kaggle login may be required). Extract the downloaded archive and copy `fer2013.csv` into this project's `data` folder so the path is `data/fer2013.csv`. CSV needs `emotion`, `pixels`, and preferably `Usage` columns. Each pixels field has 2304 values for a 48x48 grayscale face. Label order: 0 Angry, 1 Disgust, 2 Fear, 3 Happy, 4 Sad, 5 Surprise, 6 Neutral. Original splits: Training/PublicTest/PrivateTest.

## Train expression model
```bat
python train_emotion.py --csv data/fer2013.csv --epochs 25 --batch-size 64
```
For a quick pipeline test use `--epochs 1`. Model saved as `models/emotion_model.keras`. Lower batch size to 32 if memory is low. CPU training may take substantial time.

## Register identity samples
```bat
python register_face.py
```
Enter a participant name, look at webcam, vary angle/lighting gently. Up to 60 crops saved. Repeat for every consenting participant.

## Train identity model
```bat
python train_identity.py
```
Creates `models/identity_lbph.yml` and `models/identity_labels.json`.

## Run both live
```bat
python app.py
```
Press Q to quit. `app.py` uses LBPH distance threshold 75 as a starting point; lower is stricter. Tune it using separate validation captures; no universal threshold exists.

## Run the GUI
```bat
python gui.py
```
The GUI provides a live camera preview, an image-analysis button, identity registration, and buttons to train both models. Training output appears in the activity panel.

## Folder layout
```text
project/
  app.py
  register_face.py
  train_identity.py
  train_emotion.py
  check_installation.py
  requirements.txt
  data/fer2013.csv  (you download this)
  data/identity/   (created as you register people)
  models/          (trained model files created here)
```

## Troubleshooting
- `cv2.face` missing: uninstall conflicting OpenCV wheels (`pip uninstall opencv-python opencv-python-headless opencv-contrib-python-headless -y`) then `pip install opencv-contrib-python==4.10.0.84`.
- Camera unavailable: close other camera apps, allow desktop apps camera access in Windows Settings, or change `VideoCapture(0)` to `VideoCapture(1)`.
- Wrong interpreter: activate `.venv` and select it in VS Code.
- Missing model: train both models before running app.py.

## Responsible use / limitations
This is a student prototype, not secure authentication. LBPH can misidentify people with lighting, pose or camera changes. FER-2013 predicts dataset expression categories, not a person's actual internal emotional state; results can be inaccurate and biased. Obtain informed consent, secure and delete face data appropriately. Do not use this prototype for high-impact decisions or access control.
