import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import urllib.request
import pyautogui
import time
import math

# Download model file if it doesn't exist
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading hand_landmarker.task model...")
    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded!")

# Initialize hand landmarker with VIDEO running mode
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)

# Get screen dimensions for mouse control
screen_width, screen_height = pyautogui.size()

# Disable pyautogui failsafe (move mouse to corner to stop)
pyautogui.FAILSAFE = False

# Click debouncing variables
last_click_time = 0
click_debounce_time = 0.3  # 300ms between clicks
click_threshold = 0.05  # Distance threshold for click detection (increased for better detection)
clicking = False  # Track if currently clicking to prevent multiple clicks

cap = cv2.VideoCapture(0)
frame_timestamp = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Use detect_for_video (the "update method")
    detection_result = detector.detect_for_video(mp_image, frame_timestamp)
    hands = detection_result.hand_landmarks
    
    # Reset clicking state if no hands detected
    if not hands:
        clicking = False
    
    # Draw hand landmarks on the frame and control mouse
    if hands:
        h, w, _ = frame.shape
        # Use first detected hand for mouse control
        hand_landmarks = hands[0]
        
        # Get index finger tip (landmark 8) and thumb tip (landmark 4)
        index_tip = hand_landmarks[8]
        thumb_tip = hand_landmarks[4]
        
        # Map index finger position to screen coordinates and move mouse
        # Flip x-coordinate horizontally to match camera mirror view
        screen_x = int((1 - index_tip.x) * screen_width)
        screen_y = int(index_tip.y * screen_height)
        pyautogui.moveTo(screen_x, screen_y)
        
        # Calculate distance between index tip and thumb tip
        distance = math.sqrt(
            (index_tip.x - thumb_tip.x) ** 2 + 
            (index_tip.y - thumb_tip.y) ** 2
        )
        
        # Check if click should be triggered
        current_time = time.time()
        click_detected = False
        
        # Only click when fingers touch (distance < threshold) and weren't clicking before
        if distance < click_threshold:
            if not clicking and (current_time - last_click_time) > click_debounce_time:
                try:
                    pyautogui.click(button='left')
                    last_click_time = current_time
                    click_detected = True
                    clicking = True
                    print(f"Click! Distance: {distance:.4f}")
                except Exception as e:
                    print(f"Click error: {e}")
        else:
            clicking = False  # Reset clicking state when fingers separate
        
        # Display distance on frame for debugging
        cv2.putText(frame, f"Distance: {distance:.4f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Threshold: {click_threshold:.4f}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show click status
        if distance < click_threshold:
            if clicking:
                status_text = "CLICKING!"
                status_color = (0, 0, 255)  # Red
            elif (current_time - last_click_time) <= click_debounce_time:
                status_text = "Debouncing..."
                status_color = (0, 165, 255)  # Orange
            else:
                status_text = "Ready to click"
                status_color = (0, 255, 0)  # Green
        else:
            status_text = "Touch fingers to click"
            status_color = (255, 255, 255)  # White
        
        cv2.putText(frame, status_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Draw landmarks with visual feedback for click
        for idx, landmark in enumerate(hand_landmarks):
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            # Highlight index and thumb tips when click is detected
            if click_detected and (idx == 8 or idx == 4):
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)  # Red when clicking
            else:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green normally
        
        # Draw connections between landmarks
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
            (5, 9), (9, 13), (13, 17)  # Base connections
        ]
        for start_idx, end_idx in connections:
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                start = hand_landmarks[start_idx]
                end = hand_landmarks[end_idx]
                start_point = (int(start.x * w), int(start.y * h))
                end_point = (int(end.x * w), int(end.y * h))
                cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
        
        # Draw visual feedback line between index and thumb tips when close
        index_point = (int(index_tip.x * w), int(index_tip.y * h))
        thumb_point = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        if distance < click_threshold:
            cv2.line(frame, index_point, thumb_point, (0, 0, 255), 3)  # Red when clicking
        elif distance < click_threshold * 2:
            cv2.line(frame, index_point, thumb_point, (0, 255, 255), 2)  # Yellow when getting close
        
        # Draw all other hands (if multiple detected) without mouse control
        for hand_landmarks in hands[1:]:
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            
            for start_idx, end_idx in connections:
                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]
                    start_point = (int(start.x * w), int(start.y * h))
                    end_point = (int(end.x * w), int(end.y * h))
                    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
    
    cv2.imshow("virtual mouse", frame)
    
    frame_timestamp += 33  # Increment timestamp (assuming ~30 fps)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
