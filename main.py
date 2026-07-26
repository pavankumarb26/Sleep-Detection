import cv2
import numpy as np
import time
import pygame
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Step 7: Final Polishing & UI Enhancements
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Indices map for the eyes
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [263, 387, 385, 362, 380, 373]

# --- Drowsiness Logic Parameters ---
EAR_THRESHOLD = 0.20
DROWSY_DURATION = 1.0
SLEEPING_DURATION = 2.5

# State variables
eye_closed_start_time = None
current_state = "Awake"

# --- Audio Initialization ---
pygame.mixer.init()
alarm_loaded = False
alarm_playing = False

try:
    pygame.mixer.music.load("alarm.wav")
    alarm_loaded = True
    print("Successfully loaded alarm.wav.")
except pygame.error as e:
    print(f"Warning: Could not load alarm.wav. Sound will be disabled. Error: {e}")

def get_eye_coordinates(face_landmarks, indices, frame_width, frame_height):
    """Extracts the (x, y) pixel coordinates of the specified landmark indices."""
    coords = []
    for idx in indices:
        landmark = face_landmarks[idx]
        x = landmark.x * frame_width
        y = landmark.y * frame_height
        coords.append(np.array([x, y]))
    return coords

def calculate_ear(eye_coords):
    """Calculates the Eye Aspect Ratio (EAR) given 6 coordinates of an eye."""
    # Vertical distances
    d_v1 = np.linalg.norm(eye_coords[1] - eye_coords[5])
    d_v2 = np.linalg.norm(eye_coords[2] - eye_coords[4])
    
    # Horizontal distance
    d_h = np.linalg.norm(eye_coords[0] - eye_coords[3])
    
    # Calculate EAR
    ear = (d_v1 + d_v2) / (2.0 * d_h)
    return ear

# Initialize the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

print("Face Landmarker and Audio initialized. Press 'q' or 'ESC' on the camera window to exit.")

# Configure the FaceLandmarker options
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE
)

with FaceLandmarker.create_from_options(options) as landmarker:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        frame = cv2.flip(frame, 1)
        h_dim, w_dim, _ = frame.shape
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Run Face Landmarker
        result = landmarker.detect(mp_image)
        
        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]
            
            # Get coords and calculate EAR
            left_eye_coords = get_eye_coordinates(face_landmarks, LEFT_EYE_INDICES, w_dim, h_dim)
            right_eye_coords = get_eye_coordinates(face_landmarks, RIGHT_EYE_INDICES, w_dim, h_dim)
            
            left_ear = calculate_ear(left_eye_coords)
            right_ear = calculate_ear(right_eye_coords)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # --- Drowsiness State Machine ---
            if avg_ear < EAR_THRESHOLD:
                if eye_closed_start_time is None:
                    eye_closed_start_time = time.time()
                
                closed_duration = time.time() - eye_closed_start_time
                
                if closed_duration >= SLEEPING_DURATION:
                    current_state = "Sleeping - Wake Up!"
                elif closed_duration >= DROWSY_DURATION:
                    current_state = "Drowsy"
                else:
                    current_state = "Awake"
            else:
                eye_closed_start_time = None
                current_state = "Awake"
                
            # --- Alarm Sound Control Logic ---
            if current_state == "Sleeping - Wake Up!":
                if alarm_loaded and not alarm_playing:
                    try:
                        pygame.mixer.music.play(-1)
                        alarm_playing = True
                    except Exception as e:
                        print(f"Error playing sound: {e}")
            else:
                if alarm_playing:
                    try:
                        pygame.mixer.music.stop()
                        alarm_playing = False
                    except Exception as e:
                        print(f"Error stopping sound: {e}")
            
            # --- Premium UI Visualizations ---
            
            # 1. Draw Eye Outlines (Polylines)
            left_poly = np.array(left_eye_coords, dtype=np.int32)
            right_poly = np.array(right_eye_coords, dtype=np.int32)
            
            # Cyan outlines for eyes to look futuristic
            cv2.polylines(frame, [left_poly], isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            cv2.polylines(frame, [right_poly], isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            
            # 2. Draw Eye Landmark Dots
            for pt in left_eye_coords:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)  # Blue dots
            for pt in right_eye_coords:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 0, 255), -1)  # Red dots
            
            # 3. Blending a Translucent HUD Panel for Text Readability
            hud_overlay = frame.copy()
            cv2.rectangle(hud_overlay, (20, 20), (330, 160), (25, 25, 25), -1)
            # Create a 45% opaque HUD box
            cv2.addWeighted(hud_overlay, 0.45, frame, 0.55, 0, frame)
            cv2.rectangle(frame, (20, 20), (330, 160), (80, 80, 80), 1, lineType=cv2.LINE_AA)
            
            # Set state color
            if current_state == "Sleeping - Wake Up!":
                color = (0, 0, 255)      # Red for Sleeping
            elif current_state == "Drowsy":
                color = (0, 165, 255)    # Orange for Drowsy
            else:
                color = (0, 255, 0)      # Green for Awake
                
            # 4. Render Text within the HUD
            cv2.putText(
                frame, 
                f"EAR: {avg_ear:.2f}", 
                (35, 55), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.75, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )
            
            cv2.putText(
                frame, 
                f"State: {current_state}", 
                (35, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.75, 
                color, 
                2, 
                cv2.LINE_AA
            )
            
            if eye_closed_start_time is not None:
                cv2.putText(
                    frame, 
                    f"Closed for: {time.time() - eye_closed_start_time:.1f}s", 
                    (35, 135), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.75, 
                    (0, 255, 255), 
                    2, 
                    cv2.LINE_AA
                )
            
            # 5. Flashing Screen Border when User is Sleeping
            if current_state == "Sleeping - Wake Up!":
                # Flash the thick red border 3 times per second
                if int(time.time() * 6) % 2 == 0:
                    cv2.rectangle(frame, (0, 0), (w_dim, h_dim), (0, 0, 255), 18)
        
        cv2.imshow("Sleep Detection Alarm - Real-time System", frame)
        
        # Exit on 'q' or 'ESC'
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

# Clean up resources cleanly
if alarm_playing:
    pygame.mixer.music.stop()
cap.release()
cv2.destroyAllWindows()
print("Webcam and Face Landmarker resources released.")
