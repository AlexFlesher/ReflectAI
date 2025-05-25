#-------------------------------Imports--------------------------------------
import cv2
import numpy as np
#----------------------------------------------------------------------------

alpha = 0.15 # Transparency

def overlay_mask_on_frame(frame, mask):
    # Ensure mask is 2D and same height/width as frame
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    # Create an RGB mask with green and red based on in/out
    color_overlay = np.zeros_like(frame)

    # Change In-bounds color
    color_overlay[mask == 255] = (255, 255, 255)  # BGR: Green

    return cv2.addWeighted(frame, 1 - alpha, color_overlay, alpha, 0)
