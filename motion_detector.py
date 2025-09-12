import numpy as np 
import cv2  as cv
import image_filtering


def Background_Substraction(actual_img,background_img):
    grey_background_img=image_filtering.grey_scale(background_img)
    grey_actual_img=image_filtering.grey_scale(actual_img)
    abs_diff=np.abs(grey_actual_img-grey_background_img)
    return abs_diff
    
def detect_motion(abs_diff,threshold):
    return (abs_diff> threshold).astype(np.uint8) * 255   
    
    
    
def update_background(actual_img,background_img,alpha,motion_mask):
    
    np_actual=np.array(actual_img,dtype=float)
    np_background=np.array(background_img,dtype=float)
    
    # Expand mask to 3 channels if RGB
    if len(np_actual.shape) == 3 and motion_mask.ndim == 2:
        motion_mask = np.stack([motion_mask]*3, axis=-1)

    # Normalize mask to 0/1
    motion_mask = (motion_mask > 0).astype(float)
    static_mask = 1 - motion_mask
    
    new_background = np_background * motion_mask + \
                 ((1 - alpha) * np_background + alpha * np_actual) * static_mask
                 
    return new_background.astype(np.uint8)




cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
ret,background=cap.read()
if not ret :
    print("frame got captured correctly")
    cap.release()
    exit()
    
while True:
    
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    abs_diff=Background_Substraction(actual_img=frame,background_img=background)
    motion=detect_motion(abs_diff=abs_diff,threshold=15)
    kernel = np.ones((3,3), np.uint8)
    motion = cv.morphologyEx(motion, cv.MORPH_OPEN, kernel)
    motion = cv.morphologyEx(motion, cv.MORPH_DILATE, kernel)
    background=update_background(actual_img=frame,background_img=background,alpha=0.005,motion_mask=motion)
    contours, _ = cv.findContours(motion, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt) < 30:  # filter small noise
            continue
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frames
    cv.imshow("Subtractor", background)
    cv.imshow("detection", motion)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break
    else:
        continue

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()