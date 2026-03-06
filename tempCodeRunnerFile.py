import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model (nano = fastest)
model = YOLO("yolov8n.pt")

# Open video
cap = cv2.VideoCapture("videos/test_video.mp4")

if not cap.isOpened():
    print("Error: Could not open video")
    exit()

# Output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output/output_video.mp4", fourcc, 30, (416,416))

frame_count = 0

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Process only every 3rd frame (huge speed gain)
    if frame_count % 3 != 0:
        continue

    # Smaller resolution = faster
    frame = cv2.resize(frame, (416,416))

    # Fast YOLO inference
    results = model(frame, imgsz=416, conf=0.4, verbose=False)

    people_boxes = []
    people_count = 0

    safe_people = 0
    danger_people = 0

    # Detect people
    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])

            if cls == 0:

                people_count += 1

                x1,y1,x2,y2 = map(int, box.xyxy[0])
                people_boxes.append((x1,y1,x2,y2))

    danger_indices = set()

    # Fight detection
    for i in range(len(people_boxes)):
        for j in range(i+1, len(people_boxes)):

            x1,y1,x2,y2 = people_boxes[i]
            a1,b1,a2,b2 = people_boxes[j]

            center1 = ((x1+x2)//2,(y1+y2)//2)
            center2 = ((a1+a2)//2,(b1+b2)//2)

            distance = np.linalg.norm(np.array(center1)-np.array(center2))

            if distance < 60:
                danger_indices.add(i)
                danger_indices.add(j)

    # Draw boxes
    for idx,(x1,y1,x2,y2) in enumerate(people_boxes):

        if idx in danger_indices:

            danger_people += 1

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,"DANGER",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)

        else:

            safe_people += 1

            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,"SAFE",(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

    # Display info
    cv2.putText(frame,f"People: {people_count}",(20,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

    cv2.putText(frame,f"Safe: {safe_people}",(20,60),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.putText(frame,f"Danger: {danger_people}",(20,90),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    if people_count > 50:
        cv2.putText(frame,"HIGH CROWD DENSITY",(20,120),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

    out.write(frame)

    cv2.imshow("Crowd Surveillance System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()