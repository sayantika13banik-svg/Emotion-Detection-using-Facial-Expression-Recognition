import cv2
import torch
import numpy as np
from torchvision import transforms
from src.model import EmotionCNN

emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

device = torch.device('cpu')
model = EmotionCNN()

model.load_state_dict(torch.load('models/emotion_cnn.pth', map_location=device))

model.eval()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5077,), (0.2120,))
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

print("Webcam started! Press 'q' to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(48, 48)
    )

    for (x, y, w, h) in faces:
        face_crop = gray[y:y+h, x:x+w]
        input_tensor = transform(face_crop)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(device)


        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        emotion_label = emotions[predicted.item()]

        confidence_pct = confidence.item() * 100

        cv2.rectangle(
            frame,
            (x, y),           
            (x+w, y+h),       
            (0, 255, 0),      
            2                 
        )

        label_text = f"{emotion_label} ({confidence_pct:.1f}%)"

        cv2.putText(
            frame,
            label_text,       
            (x, y - 10),      
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9,              
            (0, 255, 0),      
            2                 
        )
    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")