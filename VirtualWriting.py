import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Capture video (0 = default webcam)
cap = cv2.VideoCapture(0)

# Create a blank canvas (for writing)
canvas = None

drawing = False  # writing mode
prev_x, prev_y = None, None

print("✋ Air Writing Mode")
print("Press 'W' to start/stop writing")
print("Press 'C' to clear the canvas")
print("Press 'ESC' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Camera not detected!")
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = frame.copy() * 0  # black canvas same size as frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index fingertip (landmark 8)
            index_tip = hand_landmarks.landmark[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            if drawing:
                if prev_x is not None and prev_y is not None:
                    # Draw on the canvas
                    cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 5)
                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = None, None

    # Merge the drawing on top of the camera feed
    frame = cv2.addWeighted(frame, 0.6, canvas, 0.8, 0)

    # Display writing mode
    cv2.putText(frame, f"Writing: {'ON' if drawing else 'OFF'}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0) if drawing else (0, 0, 255), 3)

    cv2.imshow("Air Writing (Press W to toggle, C to clear)", frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key in [ord('w'), ord('W')]:
        drawing = not drawing
    elif key in [ord('c'), ord('C')]:
        canvas = frame.copy() * 0  # clear the canvas

cap.release()
cv2.destroyAllWindows()
