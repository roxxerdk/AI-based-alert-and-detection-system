import cv2
from ultralytics import YOLO

# Load YOLOv8 nano model (best for CPU)
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("videos/test_video.mp4")

frame_count = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame (important for CPU)
    frame = cv2.resize(frame, (640, 480))

    frame_count += 1

    # Skip frames to improve speed
    if frame_count % 2 != 0:
        continue

    results = model(frame)

    people_count = 0

    for r in results:
        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])

            # Class 0 = person
            if cls == 0:
                people_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # Display people count
    cv2.putText(frame,
                f"People Count: {people_count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2)

    # Overcrowding alert
    if people_count > 50:
        cv2.putText(frame,
                    "WARNING: HIGH CROWD DENSITY",
                    (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

    cv2.imshow("Crowd Surveillance System", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()