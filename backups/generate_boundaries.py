import cv2
import numpy as np

# Frame Size (Post-Transform): 585 (width) x 750 (height)

def draw_boundary_zones(transformed_frame):
    zones = {
        "bottom": np.array([[0, 750], [0, 430], [585, 430], [585, 750]], dtype=np.int32), #bl, tl, tr, br
        "left": np.array([[0, 550], [0, 0], [290, 0], [145, 290]], dtype=np.int32), #bl, tl, tr, br
        "right": np.array([[620, 480], [450, 0], [640, 0], [640, 480]], dtype=np.int32), #bl, tl, tr, br
    }

    for pts in zones.values():
        overlay = transformed_frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        alpha = 0.2
        cv2.addWeighted(overlay, alpha, transformed_frame, 1 - alpha, 0, transformed_frame)

    return transformed_frame
