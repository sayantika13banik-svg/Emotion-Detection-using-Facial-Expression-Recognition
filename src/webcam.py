import cv2
import torch
import numpy as np
from torchvision import transforms
from src.model import EmotionCNN
# imports your CNN model class from src/model.py

# ── SETUP ─────────────────────────────────────────────────────

# emotion labels — must match exact order ImageFolder used
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# order must match how your model was trained
# ImageFolder sorts alphabetically — so this order is correct

# load your trained model
device = torch.device('cpu')
# using CPU here — webcam runs locally, no Colab GPU
# inference on single frames is fast enough on CPU

model = EmotionCNN()
# create empty model structure

model.load_state_dict(torch.load('models/emotion_cnn.pth', map_location=device))
# load_state_dict — loads saved weights into model
# map_location=device — loads weights onto CPU even if saved on GPU

model.eval()
# evaluation mode — turns off dropout
# always call before inference

# ── TRANSFORMS ────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.ToPILImage(),
    # cv2 reads images as numpy arrays
    # ToPILImage converts numpy array → PIL Image
    # needed because other transforms expect PIL format

    transforms.Grayscale(),
    # convert to 1 channel — model expects grayscale

    transforms.Resize((48, 48)),
    # resize face crop to 48×48 — model input size

    transforms.ToTensor(),
    # PIL Image → tensor, pixels 0-255 → 0.0-1.0

    transforms.Normalize((0.5077,), (0.2120,))
    # same normalization as training — must match exactly
])

# ── FACE DETECTOR ─────────────────────────────────────────────

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
# CascadeClassifier — pretrained face detector built into OpenCV
# haarcascades — folder of pretrained detectors
# frontalface_default — detects faces looking straight at camera
# completely separate from your CNN — just finds where face is

# ── WEBCAM LOOP ───────────────────────────────────────────────

cap = cv2.VideoCapture(0)
# VideoCapture(0) — opens default webcam
# 0 = first webcam on your system
# if you have multiple cameras, try 1 or 2

print("Webcam started! Press 'q' to quit.")

while True:
    # infinite loop — keeps reading frames until you press q

    ret, frame = cap.read()
    # cap.read() — reads one frame from webcam
    # ret = True if frame was read successfully
    # frame = the actual image as numpy array (height, width, 3)

    if not ret:
        # if frame wasn't read successfully — skip this iteration
        print("Failed to read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # convert frame to grayscale
    # why? — Haar cascade face detector works on grayscale
    # BGR = Blue Green Red — OpenCV's default color format
    # (different from RGB — Red Green Blue)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        # how much to shrink image each pass to detect different sizes
        # 1.1 = shrink by 10% each time

        minNeighbors=5,
        # how many overlapping detections needed to confirm a face
        # higher = stricter, fewer false positives

        minSize=(48, 48)
        # minimum face size to detect in pixels
        # ignores tiny false detections
    )
    # faces = list of (x, y, width, height) for each detected face
    # if no face detected → empty list

    for (x, y, w, h) in faces:
        # loop through each detected face
        # x, y = top-left corner of face rectangle
        # w, h = width and height of face rectangle

        face_crop = gray[y:y+h, x:x+w]
        # crop just the face region from grayscale frame
        # numpy array slicing: [rows, columns]
        # y:y+h = rows from y to y+h (vertical)
        # x:x+w = columns from x to x+w (horizontal)

        # ── PREDICT EMOTION ───────────────────────────────────
        input_tensor = transform(face_crop)
        # apply transforms: PIL → grayscale → 48×48 → tensor → normalize
        # output shape: (1, 48, 48)

        input_tensor = input_tensor.unsqueeze(0)
        # unsqueeze(0) — adds batch dimension
        # (1, 48, 48) → (1, 1, 48, 48)
        # model expects (batch, channels, height, width)
        # batch=1 because we're predicting one face at a time

        input_tensor = input_tensor.to(device)
        # move to CPU

        with torch.no_grad():
            # no gradient calculation needed for inference
            output = model(input_tensor)
            # forward pass — output shape (1, 7)

            probabilities = torch.softmax(output, dim=1)
            # softmax converts raw scores to probabilities
            # all 7 values sum to 1.0
            # e.g. [0.05, 0.02, 0.08, 0.72, 0.05, 0.04, 0.04]

            confidence, predicted = torch.max(probabilities, 1)
            # max probability and its index
            # confidence = 0.72 (72% sure)
            # predicted = 3 (happy)

        emotion_label = emotions[predicted.item()]
        # convert index to emotion name
        # predicted.item() converts tensor to Python int
        # emotions[3] = 'happy'

        confidence_pct = confidence.item() * 100
        # convert to percentage — 0.72 → 72.0

        # ── DRAW ON FRAME ─────────────────────────────────────
        cv2.rectangle(
            frame,
            (x, y),           # top-left corner of face box
            (x+w, y+h),       # bottom-right corner of face box
            (0, 255, 0),      # color — BGR format, (0,255,0) = green
            2                 # line thickness in pixels
        )
        # draws green rectangle around detected face

        label_text = f"{emotion_label} ({confidence_pct:.1f}%)"
        # e.g. "happy (72.3%)"

        cv2.putText(
            frame,
            label_text,       # text to display
            (x, y - 10),      # position — just above face box
            cv2.FONT_HERSHEY_SIMPLEX,  # font style
            0.9,              # font scale — size of text
            (0, 255, 0),      # color — green, matches rectangle
            2                 # thickness
        )
        # draws emotion label above the face box

    cv2.imshow('Emotion Detection', frame)
    # imshow — displays the frame in a window
    # 'Emotion Detection' — window title

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # waitKey(1) — waits 1ms for a key press
        # 0xFF — bitmask for cross-platform compatibility
        # ord('q') — ASCII code for 'q' key
        # if q is pressed → exit the loop
        break

# ── CLEANUP ───────────────────────────────────────────────────
cap.release()
# release webcam — frees up the camera for other apps

cv2.destroyAllWindows()
# closes all OpenCV windows

print("Webcam closed.")