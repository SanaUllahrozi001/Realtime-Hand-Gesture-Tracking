import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)
canvas = None  # for drawing

print(" Air Writing Mode Started!")
print("Use your index finger to write in the air.")
print(" Press 'C' to clear the screen.")
print(" Press 'ESC' to exit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get index finger tip coordinates (landmark 8)
            h, w, _ = frame.shape
            x_tip = int(hand_landmarks.landmark[8].x * w)
            y_tip = int(hand_landmarks.landmark[8].y * h)

            # Draw small circle at the fingertip
            cv2.circle(frame, (x_tip, y_tip), 8, (255, 0, 0), -1)

            # Draw line on the canvas (only when finger visible)
            if 'prev_x' in locals():
                cv2.line(canvas, (prev_x, prev_y), (x_tip, y_tip), (255, 255, 255), 5)

            prev_x, prev_y = x_tip, y_tip

    else:
        # Reset when hand not visible
        if 'prev_x' in locals():
            del prev_x, prev_y

    # Combine drawing with webcam feed
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    cv2.imshow("✍️ Air Writing", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('c') or key == ord('C'):
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()
