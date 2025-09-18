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
    
    
    
def update_background(actual_img, background_img, alpha, motion_mask):
    np_actual = actual_img.astype(np.float32)
    np_background = background_img.astype(np.float32)

    if len(np_actual.shape) == 3 and motion_mask.ndim == 2:
        motion_mask = np.stack([motion_mask] * 3, axis=-1)

    motion_mask = (motion_mask > 0).astype(np.float32)

    # Instead of blocking update completely on motion areas,
    # allow *slow* update there too
    #update_rate = (1 - motion_mask) * alpha + motion_mask * (alpha * 0.1)
    update_rate=alpha
    new_background = (1 - update_rate) * np_background + update_rate * np_actual

    return new_background.astype(np.uint8)





cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
ret, background = cap.read()
if not ret:
    print("Failed to capture background frame")
    cap.release()
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    # Preprocess
    frame_blur = cv.GaussianBlur(frame, (5,5), 0)

    abs_diff = Background_Substraction(frame_blur, background)
    motion = detect_motion(abs_diff, threshold=30)
    
    kernel = np.ones((5,5), np.uint8)
    motion = cv.morphologyEx(motion, cv.MORPH_OPEN, kernel)
    motion = cv.morphologyEx(motion, cv.MORPH_CLOSE, kernel)

    background = update_background(frame_blur, background, alpha=0.05, motion_mask=motion)
    print("Mean background value:", np.mean(background))

    contours, _ = cv.findContours(motion, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv.contourArea(cnt) < 50:  # reduce false positives
            continue
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv.imshow("Background", background)
    cv.imshow("Motion Mask", motion)
    cv.imshow("Frame", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break