#-------------------------------Imports--------------------------------------
import cv2
import numpy as np
from canvas_display import canvas_generator
#----------------------------------------------------------------------------

last_known_center = None # For handling frames without ball detection

def get_zoomed_frame(frame, center, box_size=150, output_size=(800, 600)):
    """
    Crops and zooms in on a region around the volleyball's center in the given frame.

    If the ball was not detected in the current frame, the function uses the last known center position.
    If no previous center is known, it defaults to the center of the frame.

    Parameters:
        frame (numpy.ndarray): The current video frame.
        center (tuple or int): (x, y) coordinates of the volleyball's center, or 0 if not detected.
        box_size (int): Half the side length of the square crop around the center. Default is 150.
        output_size (tuple): The size (width, height) to which the cropped region will be resized.

    Returns:
        numpy.ndarray: A zoomed-in version of the frame centered on the ball (or fallback center).
    """

    global last_known_center # Set variable to global for scope issues

    h, w, _ = frame.shape
    if center != 0:
        cx, cy = center # Get center coords
        last_known_center = center
    else: # Ball wasnt tracked
        if last_known_center == None:
            cx, cy = w // 2, h // 2 # Set center to middle of screen
        else:
            cx, cy = last_known_center

    # Define crop bounds
    left = max(cx - box_size, 0)
    right = min(cx + box_size, w)
    top = max(cy - box_size, 0)
    bottom = min(cy + box_size, h)

    # Crop frame and resize
    zoom_crop = frame[top:bottom, left:right]
    zoomed_frame = cv2.resize(zoom_crop, output_size)

    return zoomed_frame

def dim_frame(frame, factor=0.4):
    return (frame * factor).astype(np.uint8) # Dims the frame to the factor percentage, used for live feed during replays

def play_replay(replay_buffer, frame_with_mask, delay=200):
    """
    Plays a slow-motion replay sequence using the buffered frames, simulating an official review.

    The function:
    - Displays a countdown and "Review in Progress" message.
    - Plays through recent replay frames with zoom and final call annotation.
    - Renders a unified display using `canvas_generator` to combine the dimmed live feed,
      zoomed replay, and call verdict for each frame.

    Parameters:
        replay_buffer (deque): A buffer of tuples (frame, ball center, call, color) representing recent frames.
        frame_with_mask (numpy.ndarray): The current live frame with overlays, used as the dimmed background.
        delay (int): Delay in milliseconds between each replay frame. Default is 200 ms.

    Returns:
        None
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    thickness = 20

    dimmed_frame = dim_frame(frame_with_mask)

    # Show Review screen pre-messages
    for msg in ['Review in Progess!', '3', '2', '1']:

        # Show live feed (dimmed) on left side
        canvas_generator(dimmed_frame, None, None, msg)

        if cv2.waitKey(1000) == 27:  # Wait 1 second before displaying next msg, esc skips the review.
            return
        
    # Loop through all recent frames in buffer
    for frame, center, call, color in replay_buffer:
        zoomed_frame = get_zoomed_frame(frame, center)

        # Create a blank call display image
        call_display = np.full((200, 960, 3), (118,128,39), dtype=np.uint8)  # BGR format

        # Get text size to center the text
        text_size, _ = cv2.getTextSize(call, font, font_scale, thickness)
        text_x = (call_display.shape[1] - text_size[0]) // 2 + 10 # Custom parameters for centering
        text_y = (call_display.shape[0] + text_size[1]) // 2 - 40

        # Draw call text on the call display
        cv2.putText(call_display, call, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

        # Show canvas with live feed, replay and call displays
        canvas_generator(dimmed_frame, zoomed_frame, call_display)

        if cv2.waitKey(delay) == 27:  # ESC exits replay
            break

    cv2.destroyAllWindows() # End replay once all frames are looped through
