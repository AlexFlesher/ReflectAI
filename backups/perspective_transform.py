#--------------------------------------------------------------------------------#

# NOTE: This is no longer in use for the final project.

#--------------------------------------------------------------------------------#
import cv2
import numpy as np

# Read Video File
#-------------------------------------------------------------------------------#
cap = cv2.VideoCapture('videos/raw_gameplay/empty_pool_converted.mp4')
success, image = cap.read()
cv2.imwrite('frame_check.jpg', image)
#-------------------------------------------------------------------------------#

# Coordinates for perspective transform
tl = (80, 0)
bl = (80, 550)
tr = (580, 0)
br = (580, 550)

while success:
    success, image = cap.read()
    if not success or image is None:
        break  # Exit loop if frame is empty

    # Edit frame size
    frame = image[:, 665:1250]

    # Draw Boundary Dots
    for pt in [tl, bl, tr, br]:
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)

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

    # Show Transformed Frames
    cv2.imshow('Frame', frame)
    cv2.imshow('Transformed Frame', transformed_frame)

    # Frame Delay
    if cv2.waitKey(50)==27: # Esc key, waitKey(ms)
        break
