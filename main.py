import cv2
import numpy as np
import os
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

video_folder = "videos"
os.makedirs("output", exist_ok=True)

video_files = os.listdir(video_folder)

for video_name in video_files:

    video_path = os.path.join(video_folder, video_name)
    print("Processing:", video_name)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video:", video_name)
        continue

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f"output/output_{video_name}", fourcc, 25, (640,480))

    frame_count = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame,(640,480))
        frame_count += 1

        # Skip frames for faster processing
        if frame_count % 3 != 0:
            out.write(frame)
            continue

        ####################################################
        # PEOPLE DETECTION
        ####################################################

        if any(x in video_name.lower() for x in ["abnormal","fight","crowd"]):

            results = model(frame, imgsz=640, conf=0.5, verbose=False)

            people_boxes = []
            centers = []

            for r in results:
                for box in r.boxes:

                    cls = int(box.cls[0])

                    # detect only PERSON
                    if cls == 0:

                        x1,y1,x2,y2 = map(int,box.xyxy[0])
                        people_boxes.append((x1,y1,x2,y2))

                        cx = (x1+x2)//2
                        cy = (y1+y2)//2
                        centers.append((cx,cy))

            people_count = len(people_boxes)

            ####################################################
            # CLASH DETECTION
            ####################################################

            danger_indices = set()

            for i in range(len(centers)):
                for j in range(i+1,len(centers)):

                    dist = np.linalg.norm(
                        np.array(centers[i]) - np.array(centers[j])
                    )

                    if dist < 60:
                        danger_indices.add(i)
                        danger_indices.add(j)

            safe_count = 0
            danger_count = 0

            ####################################################
            # DRAW BOXES
            ####################################################

            for i,(x1,y1,x2,y2) in enumerate(people_boxes):

                if i in danger_indices:

                    danger_count += 1

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

                    cv2.putText(frame,"DANGER",
                                (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,0,255),2)

                else:

                    safe_count += 1

                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                    cv2.putText(frame,"SAFE",
                                (x1,y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,(0,255,0),2)

            ####################################################
            # DISPLAY COUNTS
            ####################################################

            cv2.putText(frame,f"People: {people_count}",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,(255,255,255),2)

            cv2.putText(frame,f"Safe: {safe_count}",
                        (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(0,255,0),2)

            cv2.putText(frame,f"Danger: {danger_count}",
                        (20,110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(0,0,255),2)

        ####################################################
        # FIRE DETECTION
        ####################################################

        elif "fire" in video_name.lower():

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Fire color ranges
            lower_fire1 = np.array([0,120,200])
            upper_fire1 = np.array([35,255,255])

            lower_fire2 = np.array([160,120,200])
            upper_fire2 = np.array([180,255,255])

            mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
            mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)

            fire_mask = cv2.bitwise_or(mask1, mask2)

            kernel = np.ones((5,5),np.uint8)

            fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
            fire_mask = cv2.dilate(fire_mask, kernel, iterations=2)

            contours,_ = cv2.findContours(
                fire_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:

                if cv2.contourArea(cnt) > 400:

                    x,y,w,h = cv2.boundingRect(cnt)

                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

                    cv2.putText(frame,
                                "FIRE DANGER",
                                (x,y-10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0,0,255),
                                2)

            cv2.putText(frame,
                        "ALERT: FIRE DETECTED",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3)

        ####################################################

        out.write(frame)

        cv2.imshow("AI Surveillance System", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()

cv2.destroyAllWindows()