import cv2
import mediapipe as mp
import numpy as np
from collections import Counter, deque

# Initialize MediaPipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

# Store predictions for smoothing
gesture_predictions = deque(maxlen=7)  # Reduced from 10 to 7 for faster response

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_angle(a, b, c):
    """Calculate angle between three points (in degrees)"""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

def is_finger_extended(landmarks, finger_tip_idx, finger_pip_idx, wrist_idx):
    """Check if a finger is extended"""
    tip = landmarks[finger_tip_idx]
    pip = landmarks[finger_pip_idx]
    wrist = landmarks[wrist_idx]
    
    # Distance from tip to wrist should be greater than pip to wrist
    tip_to_wrist = calculate_distance(tip, wrist)
    pip_to_wrist = calculate_distance(pip, wrist)
    
    # Made more lenient: reduced threshold from 1.1 to 1.05
    return tip_to_wrist > pip_to_wrist * 1.05

def detect_hello(landmarks):
    """
    Detect 'Hello' gesture:
    - Open palm with fingers SPREAD APART (key difference!)
    - Hand should be relatively upright
    - Fingers separated (waving position)
    """
    wrist = landmarks[0]
    
    # Check if all fingers are extended
    fingers_extended = [
        is_finger_extended(landmarks, 8, 6, 0),   # Index
        is_finger_extended(landmarks, 12, 10, 0), # Middle
        is_finger_extended(landmarks, 16, 14, 0), # Ring
        is_finger_extended(landmarks, 20, 18, 0)  # Pinky
    ]
    
    # Check thumb extension (different logic)
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    thumb_extended = calculate_distance(thumb_tip, wrist) > calculate_distance(thumb_mcp, wrist) * 1.1
    
    # At least 3 fingers should be extended (more lenient)
    most_extended = sum(fingers_extended) >= 3 and thumb_extended
    
    # Check hand orientation (palm should be facing forward/camera)
    middle_tip = landmarks[12]
    middle_mcp = landmarks[9]
    
    # Hand should be upright (middle finger tip above mcp) - more lenient
    upright = middle_tip[1] < middle_mcp[1] + 0.05
    
    # KEY DIFFERENCE FOR HELLO: Fingers MUST be SPREAD APART
    hand_size = calculate_distance(landmarks[0], landmarks[9])
    index_middle_dist = calculate_distance(landmarks[8], landmarks[12])
    middle_ring_dist = calculate_distance(landmarks[12], landmarks[16])
    ring_pinky_dist = calculate_distance(landmarks[16], landmarks[20])
    
    # Check multiple finger gaps to ensure spreading
    # At least 2 gaps should be wide
    gap_threshold = hand_size * 0.28
    wide_gaps = sum([
        index_middle_dist > gap_threshold,
        middle_ring_dist > gap_threshold,
        ring_pinky_dist > gap_threshold
    ])
    
    fingers_spread = wide_gaps >= 2
    
    # Hello can be anywhere from mid-frame downward (not at chin)
    # Chin level is typically 0.15-0.4, so Hello should be below that
    not_at_chin = middle_tip[1] > 0.35  # Can be higher now, just not at chin
    
    return most_extended and upright and fingers_spread and not_at_chin

def detect_thank_you(landmarks):
    """
    Detect 'Thank You' gesture (ASL):
    - Open hand with fingers CLOSE TOGETHER (key difference!)
    - Fingertips touch or near chin/lips
    - All fingers extended but NOT spread apart
    """
    wrist = landmarks[0]
    middle_tip = landmarks[12]
    index_tip = landmarks[8]
    
    # Check if all fingers are extended
    fingers_extended = [
        is_finger_extended(landmarks, 8, 6, 0),
        is_finger_extended(landmarks, 12, 10, 0),
        is_finger_extended(landmarks, 16, 14, 0),
        is_finger_extended(landmarks, 20, 18, 0)
    ]
    
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    thumb_extended = calculate_distance(thumb_tip, wrist) > calculate_distance(thumb_mcp, wrist) * 1.1
    
    # At least 3 fingers extended (more lenient)
    most_extended = sum(fingers_extended) >= 3 and thumb_extended
    
    # KEY DIFFERENCE FOR THANK YOU: Fingers MUST be CLOSE TOGETHER
    # Check if fingers are grouped together (not spread apart)
    index_middle_dist = calculate_distance(landmarks[8], landmarks[12])
    middle_ring_dist = calculate_distance(landmarks[12], landmarks[16])
    ring_pinky_dist = calculate_distance(landmarks[16], landmarks[20])
    
    # Normalize distances relative to hand size
    hand_size = calculate_distance(landmarks[0], landmarks[9])  # wrist to middle knuckle
    
    # ALL gaps must be small for Thank You
    gap_threshold = hand_size * 0.35
    all_gaps_small = (
        index_middle_dist < gap_threshold and
        middle_ring_dist < gap_threshold and
        ring_pinky_dist < gap_threshold
    )
    
    fingers_together = all_gaps_small
    
    # Hand should be in upper-middle portion of frame (chin/face level)
    # Y coordinate should be in upper 50% (lower values = higher position)
    # More lenient range
    at_chin_level = 0.15 < middle_tip[1] < 0.6
    
    # Palm orientation: fingers pointing slightly upward or forward
    # Fingertips should be at similar height or slightly above knuckles
    middle_mcp = landmarks[9]
    palm_facing_out = middle_tip[1] <= middle_mcp[1] + 0.15
    
    return most_extended and fingers_together and at_chin_level and palm_facing_out

def detect_yes(landmarks):
    """
    Detect 'Yes' gesture:
    - Closed fist (all fingers curled)
    - OR pointing gesture (index extended, others closed)
    """
    wrist = landmarks[0]
    
    # Check which fingers are extended
    index_extended = is_finger_extended(landmarks, 8, 6, 0)
    middle_extended = is_finger_extended(landmarks, 12, 10, 0)
    ring_extended = is_finger_extended(landmarks, 16, 14, 0)
    pinky_extended = is_finger_extended(landmarks, 20, 18, 0)
    
    # For "Yes" - fist gesture (no fingers or max 1 finger extended)
    fist = sum([index_extended, middle_extended, ring_extended, pinky_extended]) <= 1
    
    # Alternative: pointing gesture (only index extended)
    pointing = index_extended and not any([middle_extended, ring_extended, pinky_extended])
    
    return fist or pointing

def detect_gesture(landmarks):
    """Main gesture detection function - order matters!"""
    # Convert normalized coordinates to pixel coordinates
    # Assuming landmarks are already in pixel format
    
    # Check Thank You FIRST (more specific gesture at chin level)
    if detect_thank_you(landmarks):
        return "Thank You"
    # Then check Hello (spread fingers, lower position)
    elif detect_hello(landmarks):
        return "Hello"
    # Finally check Yes
    elif detect_yes(landmarks):
        return "Yes"
    else:
        return "Unknown"

def filter_predictions(new_prediction):
    """Apply temporal smoothing using majority voting"""
    gesture_predictions.append(new_prediction)
    
    # Get most common prediction
    if len(gesture_predictions) > 0:
        most_common = Counter(gesture_predictions).most_common(1)[0][0]
        return most_common
    return new_prediction

# Main loop
print("Sign Language Recognition Started!")
print("Gestures: Hello | Thank You | Yes")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks, 
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # Get landmark positions (normalized 0-1)
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            
            # Detect gesture
            gesture = detect_gesture(landmarks)
            
            # Apply smoothing
            gesture = filter_predictions(gesture)
            
            # Display gesture
            color = (0, 255, 0) if gesture != "Unknown" else (0, 0, 255)
            cv2.putText(frame, f"Gesture: {gesture}", (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Display instructions
            cv2.putText(frame, "Hello: SPREAD fingers apart, chest/shoulder level", (10, h-90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Thank You: Fingers TOGETHER at chin level", (10, h-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "Yes: Closed fist or pointing", (10, h-30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        cv2.putText(frame, "No hand detected", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show frame
    cv2.imshow("Sign Language Recognition", frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()