#-------------------------------Imports--------------------------------------
import cv2
#----------------------------------------------------------------------------

def get_ball_position(results): # Get the coordinates of the volleyball
    """
    Extracts the center coordinates of the detected volleyball from YOLO model results.

    Parameters:
        results (list): Output from the YOLO model prediction, containing bounding boxes and class IDs.

    Returns:
        tuple or int: (x, y) coordinates of the center of the detected volleyball as integers.
                      Returns 0 if no volleyball is found.
    """

    # Get Object Tensors
    bounding_boxes = results[0].boxes

    # Convert tensors to numpy arrays for iteration
    coords = bounding_boxes.xyxy.cpu().numpy() # Top Left X Y and Bottom Right X Y Coords
    classID = bounding_boxes.cls.cpu().numpy().astype(int) # Volleyball (0) or Player (1)

    volleyball_center = 0 # Initialize
    for i in range(len(coords)): 
        if classID[i] == 0:  # If object is volleyball
            x1, y1, x2, y2 = coords[i]
            center_x = (x1 + x2) / 2 # Find center of box
            center_y = (y1 + y2) / 2
            volleyball_center = (int(center_x), int(center_y))
            
    return volleyball_center
    #----------------------------------------------------------------------------------------


def get_call(mask, volleyball_center):
    """
    Determines whether the volleyball is 'IN' or 'OUT' based on its position relative to a binary mask.

    Parameters:
        mask (numpy.ndarray): Grayscale image representing the court bounds, where white indicates in-bounds areas.
        volleyball_center (tuple or int): (x, y) coordinates of the volleyball's center, or 0 if not detected.

    Returns:
        tuple: 
            - call (str): 'IN' if the ball is within the white area of the mask, 'OUT' if outside, or '' if undetected.
            - color (tuple): BGR color tuple representing the call (green for IN, purple for OUT, black for no call).
    """
    
    if volleyball_center != 0:
        x, y = volleyball_center[0], volleyball_center[1]
    else:
        x = y = 0 # Frame didnt track volleyball
        
    # Convert mask to binary thresh
    binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)[1] # Convert mask to purely black and white (If pixel value is > 127, make it white, if not, make it black)

    # Get height and width of binary mask
    h, w = binary_mask.shape

    # Check if volleyball is inside white colored mask
    if 0 <= x < w and 0 <= y <= h and volleyball_center != 0:
        if binary_mask[y, x] == 1: # Inside white area
            call = 'IN'
            color = (55, 255, 0)
        else: # Not inside white area
            call = 'OUT'
            color = (138, 148, 241)
    else:
        call = '' # Frame didnt track volleyball
        color = (0, 0, 0) # None

    return call, color