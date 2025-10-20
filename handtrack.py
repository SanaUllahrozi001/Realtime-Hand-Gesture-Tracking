import cv2
import mediapipe as mp

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Start webcam
cap = cv2.VideoCapture(0)
print("🚀 Starting Advanced Hand Gesture Recognition...")
print("Press 'ESC' to exit")

# Finger landmark indices
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_BASES = [3, 6, 10, 14, 18]

def get_finger_status(hand_landmarks):
    finger_status = []
    landmarks = hand_landmarks.landmark

    # Thumb (check left or right hand)
    if landmarks[FINGER_TIPS[0]].x < landmarks[FINGER_BASES[0]].x:
        finger_status.append(1)
    else:
        finger_status.append(0)

    # Other four fingers
    for i in range(1, 5):
        if landmarks[FINGER_TIPS[i]].y < landmarks[FINGER_BASES[i]].y:
            finger_status.append(1)
        else:
            finger_status.append(0)
    return finger_status

def recognize_gesture(fingers):
    # Define gestures based on finger pattern
    if fingers == [0, 1, 0, 0, 0]:
        return "☝️ Pointing Up"
    elif fingers == [0, 1, 1, 0, 0]:
        return "✌️ Victory"
    elif fingers == [1, 0, 0, 0, 0]:
        return "👍 Thumbs Up"
    elif fingers == [0, 0, 0, 0, 0]:
        return "✊ Fist"
    elif fingers == [1, 1, 1, 1, 1]:
        return "🖐️ Open Hand"
    elif fingers == [0, 0, 0, 0, 1]:
        return "🤙 Call Me"
    elif fingers == [1, 0, 0, 0, 1]:
        return "🤘 Rock Sign"
    elif fingers == [0, 1, 1, 1, 1]:
        return "✋ Stop"
    elif fingers == [1, 0, 0, 0, 0] and thumb_down():
        return "👎 Thumbs Down"
    else:
        return ""

def thumb_down():
    # Placeholder logic for thumbs down
    return False  # can be enhanced later

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("❌ Camera not detected!")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers = get_finger_status(hand_landmarks)
            gesture = recognize_gesture(fingers)

            if gesture:
                cv2.putText(frame, gesture, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.5, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow("Advanced Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
