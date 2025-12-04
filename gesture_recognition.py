# import cv2
# import mediapipe as mp

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
# mp_draw = mp.solutions.drawing_utils

# # Start webcam capture
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip the frame horizontally for selfie-view display
#     frame = cv2.flip(frame, 1)

#     # Convert the BGR frame to RGB for MediaPipe processing
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process the frame and detect hands
#     results = hands.process(rgb_frame)

#     # If hands are detected, draw landmarks
#     if results.multi_hand_landmarks:
#         for landmarks in results.multi_hand_landmarks:
#             mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

#     # Display the frame with landmarks
#     cv2.imshow("Hand Tracking", frame)

#     # Press 'q' to quit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import math
from collections import Counter

# Initialize MediaPipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture (webcam)
cap = cv2.VideoCapture(0)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    # Calculate the angle between three points a, b, c using the dot product
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])

    dot_product = np.dot(ba, bc)
    magnitude_ba = np.linalg.norm(ba)
    magnitude_bc = np.linalg.norm(bc)
    
    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)  # Return angle in degrees

# Gesture Recognition using angle-based features
def detect_gesture(landmarks):
    # Extract the angles between key points (thumb, index, etc.)
    # Example: Calculate angle between thumb, index, and middle fingers
    
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    
    # Calculate angles between thumb and index, and index and middle
    thumb_index_angle = calculate_angle(thumb_tip, landmarks[3], index_tip)
    index_middle_angle = calculate_angle(index_tip, landmarks[7], middle_tip)
    
    # Define threshold values for gesture classification
    if thumb_index_angle < 50 and index_middle_angle < 50:
        return "Thumbs Up"
    elif thumb_index_angle > 140 and index_middle_angle > 140:
        return "Fist"
    else:
        return "Open Hand"

# Post-processing with majority voting for gesture smoothing
def filter_predictions(new_prediction):
    global gesture_predictions
    
    # Append the new prediction to the list
    gesture_predictions.append(new_prediction)
    
    # Keep the last 5 predictions for voting
    if len(gesture_predictions) > 5:
        gesture_predictions.pop(0)
    
    # Perform majority voting
    most_common_gesture = Counter(gesture_predictions).most_common(1)[0][0]
    return most_common_gesture

# Store previous gesture predictions for post-processing
gesture_predictions = []

# Loop through the webcam feed
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame for a mirror view
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get landmark positions
            landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in hand_landmarks.landmark]

            # Draw landmarks and connections
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Detect gesture based on landmarks
            gesture = detect_gesture(landmarks)
            
            # Apply post-processing (majority voting)
            gesture = filter_predictions(gesture)
            
            # Display the gesture on screen
            cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Show the frame with the detected gestures
    cv2.imshow("Gesture Recognition", frame)
    
    # Exit the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
