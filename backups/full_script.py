import cv2
import numpy as np

# Read Video File
#-------------------------------------------------------------------------------#
cap = cv2.VideoCapture('videos/raw_gameplay/empty_pool_converted.mp4')
success, image = cap.read()
cv2.imwrite('frame_check.jpg', image)
#-------------------------------------------------------------------------------#

while success:
    success, image = cap.read()
    if not success or image is None:
        break  # Exit loop if frame is empty
    
    # Edit frame size
    frame = image[200:800, 665:1250]

    # Select Coordinates
    tl = (0, 0)
    bl = (50, 350)
    tr = (580, 0)
    br = (580, 350)

    # Draw Boundrary Dots
    cv2.circle(frame, tl, 5, (0, 0, 255), -1)
    cv2.circle(frame, bl, 5, (0, 0, 255), -1)
    cv2.circle(frame, tr, 5, (0, 0, 255), -1)
    cv2.circle(frame, br, 5, (0, 0, 255), -1)

    # Perspective Transformation
    #-------------------------------------------------------------------------------#
    pts1 = np.float32([tl, bl, tr, br])
    pts2 = np.float32([
        [0, 0], 
        [0, 480], 
        [640, 0], 
        [640, 480]
    ])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, (640, 480))
    #-------------------------------------------------------------------------------#

    # Boundrary Boxes (Without the Top Horizontal Box, that will be handles seperatley)
    #-------------------------------------------------------------------------------#
    zones = {
        "bottom": np.array([[0, 480], [0, 305], [640, 315], [640, 480]], dtype=np.int32), #bl, tl, tr, br
        "left": np.array([[0, 150], [0, 0], [220, 0], [145, 140]], dtype=np.int32), #bl, tl, tr, br
        "right": np.array([[620, 480], [450, 0], [640, 0], [640, 480]], dtype=np.int32), #bl, tl, tr, br
    }

    for pts in zones.values():
        overlay = transformed_frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 255))
        cv2.addWeighted(overlay, 0.4, transformed_frame, 0.6, 0, transformed_frame)

    #-------------------------------------------------------------------------------#

    # Show Transformed Frames
    cv2.imshow('Frame', frame)
    cv2.imshow('Transformed Frame', transformed_frame)

    # Frame Delay
    if cv2.waitKey(50)==27: # Esc key, waitKey(ms)
        break