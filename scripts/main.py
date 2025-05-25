#-------------------------------Imports--------------------------------------
import cv2
import time
from ultralytics import YOLO
from collections import deque
from multiprocessing import Event
from mask_utils import overlay_mask_on_frame
from ball_utils import get_ball_position
from ball_utils import get_call
from replay_manager import play_replay
from canvas_display import canvas_generator
from challenge_listener import start_audio_listener
#----------------------------------------------------------------------------

model = YOLO("models/yolov8n_trained2.pt") # Select the model to run on video

cap = cv2.VideoCapture(0) # Select the video to analyze

# Force HD resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

success, image = cap.read() # Reads the first frame

mask = cv2.imread('dataset/bounds_mask.jpg', cv2.IMREAD_GRAYSCALE) # Select the overlay mask path

# Replay Settings
FPS = 30
REPLAY_DURATION = 5  # seconds
MAX_BUFFER_SIZE = FPS * REPLAY_DURATION
replay_buffer = deque(maxlen=MAX_BUFFER_SIZE)

# Save First Frame to Directory
if success and image is not None:
    frame = image[:750, 665:1250]

    # Save the cropped frame
    cv2.imwrite("first_frame.jpg", frame)

#--------------------------Initialize Audio Detection--------------------------
# Initialize trigger
challenge_event = Event()

# Path to your .ppn model file
keyword_path = "models/challenge_en_windows.ppn"  # Adjust path as needed

# Start the listener
start_audio_listener(challenge_event, keyword_path)
#------------------------------------------------------------------------------

# Start main loop through all frames to generate the live video
while True:

    start_time = time.time() # For FPS and Lag UI

    success, image = cap.read() # Read frame
    if not success or image is None:
        break
    
    # Crop image (remove black bars from portrait recording)
    frame = image.copy() # For other resolution testing
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = frame[200:1400, :]

    print(image.shape)

    # Run and show YOLO Model Tracking on the frame
    results = model.predict(frame, verbose=False, conf=0.1)
    YOLO_frame = results[0].plot()

    # Show mask on the frame
    YOLO_mask_frame = overlay_mask_on_frame(YOLO_frame, mask)

    #-------------------------------Volleyball Tracking--------------------------------------
    # Get coords of volleyball
    volleyball_center = get_ball_position(results)
    if volleyball_center != 0:
        x, y = volleyball_center[0], volleyball_center[1]
    else:
        x = y = 0 # Fail case

    # Determine if volleyball is in or out
    call, color = get_call(mask, volleyball_center)
    #----------------------------------------------------------------------------------------

    # Save frames for future potential challenges
    replay_buffer.append((YOLO_mask_frame.copy(), volleyball_center, call, color))

    # Change frame name with new UI
    YOLO_mask_UI_frame = YOLO_mask_frame

    # Determine FPS and Lag
    elapsed = time.time() - start_time
    fps = 1 / elapsed if elapsed > 0 else 0
    lag = elapsed * 1000
    cv2.putText(YOLO_mask_UI_frame, f"FPS: {fps:.1f} | Lag: {lag:.1f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Put 'AquaCam' Title on Live Frame
    cv2.putText(YOLO_mask_UI_frame, f"AquaCam | Reflect AI v1.0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (224, 187, 54), 2, cv2.LINE_AA)

    #-------------------------------Challenge/Replay Check--------------------------------------
    key = cv2.waitKey(1)

    if key == 27:  # ESC key, exit loop
        break
    elif key == ord('c') or challenge_event.is_set(): # If 'Challenge" audio is deteched
        play_replay(replay_buffer, YOLO_mask_UI_frame)
        challenge_event.clear() # Reset for next call
    #-------------------------------------------------------------------------------------------
    
    # Show call on frame
    if call == 'IN':
        color = (60, 150, 60)

    # Change frame new for final changes
    final_frame = YOLO_mask_UI_frame

    cv2.putText(final_frame, call, (x + 25, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 4, cv2.LINE_AA)

    # Create Windows, Configure, and Show Them.
    #---------------------------------------------------------------------------------------------------------------------
    canvas_generator(final_frame)
    #---------------------------------------------------------------------------------------------------------------------

# End capturing resources and close windows
cap.release()
cv2.destroyAllWindows()