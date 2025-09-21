import cv2
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO('yolov8n.pt')

# Open webcam (0) or video file
cap = cv2.VideoCapture(0)  # change to "your_video.mp4" if needed
threshold = 20
tracked_objects = {}

while cap.isOpened():
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection + tracking
    results = model.track(frame, persist=True,show=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            obj_id = int(box.id.item()) if box.id is not None else None
            current_centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

            if obj_id is not None and obj_id in tracked_objects:
                prev_centroid = tracked_objects[obj_id]
                displacement_x = abs(current_centroid[0] - prev_centroid[0])
                displacement_y = abs(current_centroid[1] - prev_centroid[1])
                if displacement_x > threshold or displacement_y > threshold:
                    print(f"Motion detected for object {obj_id}!")

            if obj_id is not None:
                tracked_objects[obj_id] = current_centroid

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if obj_id is not None:
                cv2.putText(frame, f"ID: {obj_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Calculate FPS
    fps = 1 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show real-time video
    cv2.imshow("Real-Time YOLO Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
