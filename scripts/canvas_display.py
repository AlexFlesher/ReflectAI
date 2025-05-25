#-------------------------------Imports--------------------------------------
import cv2
import numpy as np
import time
#----------------------------------------------------------------------------

def add_border(img, border_size=10, color=(255, 255, 255)):
    return cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=color) # Add white border to frame

def canvas_generator(live_frame, replay_frame=None, call_display=None, text='Waiting for Challenge ...'):
    """
    Renders the full-screen canvas for the application, displaying either the live feed or replay UI.

    Depending on the inputs:
    - If `replay_frame` and `call_display` are provided, it renders the replay view with titles and verdict.
    - Otherwise, it shows the live feed on the left and a centered animated status message on the right.

    Parameters:
        live_frame (numpy.ndarray): The live video frame to display on the left side of the canvas.
        replay_frame (numpy.ndarray, optional): The zoomed-in replay frame to show fullscreen if present.
        call_display (numpy.ndarray, optional): The verdict display image (e.g., "IN" or "OUT").
        text (str): Text to display when not in replay mode. Supports animation with ellipses.

    Returns:
        None
    """
    
    window_name = "Full Display"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Make window fullscreen

    canvas = np.full((1080, 1920, 3), (118,128,39), dtype=np.uint8)  # BGR format

    # ----------Live Frame (Left Side)----------
    live_resized = cv2.resize(live_frame, (940, 1060))
    live_with_border = add_border(live_resized)
    canvas[:, :960] = live_with_border
    #-------------------------------------------

    if replay_frame is not None and call_display is not None: # Replay is showing
        #-------------------Replay Text Label-------------------
        canvas[:, :960] = (118,128,39)
        text = "Replay"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.8
        thickness = 6
        size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (1920 - size[0]) // 2
        y = 90
        cv2.putText(canvas, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        #-------------------------------------------------------

        #---------------------Replay Window---------------------
        replay_window_height = 720
        replay_resized = cv2.resize(replay_frame, (1920, replay_window_height))
        canvas[120:120 + replay_window_height, 0:1920] = replay_resized
        #-------------------------------------------------------

        #---------------------Final Call Label---------------------
        text = "Final Call"
        font_scale = 1.5
        thickness = 6
        size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (1920 - size[0]) // 2
        y = 120 + replay_window_height + 45  # some padding below the replay window
        cv2.putText(canvas, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        #----------------------------------------------------------

        #---------------------Final Call Display---------------------
        call_display_height = 180
        call_resized = cv2.resize(call_display, (1920, call_display_height))
        canvas[1080 - call_display_height:1080, 0:1920] = call_resized
        #------------------------------------------------------------

    else: # No active replay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 5
        thickness = 15
        center_y = 580
        line_height = 200

        # Animate ellipses based on time
        dot_count = int(time.time() * 2) % 4
        if '.' in text:
            text = "Waiting for Challenge " + "." * dot_count
            center_y += 80

        lines = text.split(' ')
        base_y = center_y - (len(lines) - 1) * line_height // 2 # Starting y height for first line

        for i, word in enumerate(lines):
            y = base_y + i * line_height # Increase y for each new word
            if '.' in word:
                y -= 80 # Less line height between last line and elipses

            size, _ = cv2.getTextSize(word, font, font_scale, thickness)
            x = 970 + (960 - size[0]) // 2 # Center x for word size
            cv2.putText(canvas, word, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


    cv2.imshow(window_name, canvas)
