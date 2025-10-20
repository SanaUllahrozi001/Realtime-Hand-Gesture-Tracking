import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 for object + human detection
yolo_model = YOLO('yolov8n.pt')

# MediaPipe setup for hand detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)

# Load pretrained age & gender models (from OpenCV’s DNN)
age_proto = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/AgeGender/models/age_deploy.prototxt"
age_model = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/AgeGender/models/age_net.caffemodel"
gender_proto = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/AgeGender/models/gender_deploy.prototxt"
gender_model = "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/AgeGender/models/gender_net.caffemodel"

# Load DNN models
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)
gender_net = cv2.dnn.readNetFromCaffe(gender_proto, gender_model)

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Object & human detection (YOLO)
    results = yolo_model(frame)
    annotated_frame = results[0].plot()

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Hand Detection ---
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_draw.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.putText(annotated_frame, "Hand Detected", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- Face + Age + Gender Detection ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.426, 87.769, 114.896), swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]

        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_LIST[age_preds[0].argmax()]

        overlay_text = f"{gender}, {age}"
        cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(annotated_frame, overlay_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # --- Display final frame ---
    cv2.putText(annotated_frame, "Press Q to Quit", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
