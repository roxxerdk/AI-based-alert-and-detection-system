import cv2
from ultralytics import YOLO

# load model once
model = YOLO("yolov8n.pt")

def detect_victims(frame):

    victims = []

    results = model(frame, conf=0.4, verbose=False)

    for r in results:
        for box in r.boxes:

            cls = int(box.cls[0])

            # class 0 = person
            if cls == 0:

                x1,y1,x2,y2 = map(int, box.xyxy[0])

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                victims.append((cx,cy))

    return victims