import cv2
import mediapipe as mp
import numpy as np
from collections import Counter, deque

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])

    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def finger_extended(landmarks, mcp_idx, pip_idx, dip_idx, tip_idx, threshold=160):
    mcp = landmarks[mcp_idx]
    pip = landmarks[pip_idx]
    dip = landmarks[dip_idx]
    tip = landmarks[tip_idx]
    angle = calculate_angle(mcp, pip, dip)
    return angle > threshold and tip[1] < mcp[1]

def thumb_extended(landmarks, threshold=150):
    cmc = landmarks[1]
    mcp = landmarks[2]
    tip = landmarks[4]
    angle = calculate_angle(cmc, mcp, tip)
    return angle > threshold and tip[0] < mcp[0]

def get_finger_states(landmarks):
    return {
        "thumb": thumb_extended(landmarks),
        "index": finger_extended(landmarks, 5, 6, 7, 8),
        "middle": finger_extended(landmarks, 9, 10, 11, 12),
        "ring": finger_extended(landmarks, 13, 14, 15, 16),
        "pinky": finger_extended(landmarks, 17, 18, 19, 20),
    }

gesture_predictions = []
wrist_history = deque(maxlen=20)

def detect_gesture(landmarks):
    states = get_finger_states(landmarks)
    extended_count = sum(states.values())

    fist = extended_count == 0
    open_hand = extended_count >= 4

    waving = False
    stable = False
    if len(wrist_history) >= 8:
        xs = [p[0] for p in wrist_history]
        spread = max(xs) - min(xs)
        direction_changes = 0
        last_dx = 0
        for i in range(1, len(xs)):
            dx = xs[i] - xs[i - 1]
            if abs(dx) < 3:
                continue
            if last_dx == 0:
                last_dx = dx
                continue
            if np.sign(dx) != np.sign(last_dx):
                direction_changes += 1
                last_dx = dx
        waving = spread > 90 and direction_changes >= 3
        stable = spread < 25 and direction_changes == 0

    if fist:
        return "Yes"
    if open_hand and waving:
        return "Hello"
    if open_hand and not waving and stable:
        return "Thank You"
    return "Unknown"

def filter_predictions(new_prediction):
    gesture_predictions.append(new_prediction)
    if len(gesture_predictions) > 7:
        gesture_predictions.pop(0)
    return Counter(gesture_predictions).most_common(1)[0][0]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]

            wrist_history.append(landmarks[0])

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            gesture = detect_gesture(landmarks)
            gesture = filter_predictions(gesture)

            cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    else:
        wrist_history.clear()

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
