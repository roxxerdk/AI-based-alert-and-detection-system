
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

tracker = DeepSort(max_age=20)

video_folder = "videos"

video_list = [
    os.path.join(video_folder, v)
    for v in os.listdir(video_folder)
    if v.endswith(".mp4")
]

for video_path in video_list:

    print("Processing:", video_path)

    cap = cv2.VideoCapture(video_path)

    video_name = os.path.basename(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(
        f"output/output_{video_name}",
        fourcc,
        30,
        (640,640)
    )

    frame_count = 0

    is_people_video = any(
        x in video_name.lower()
        for x in ["fight","crowd","abnormal"]
    )

    is_fire_video = "fire" in video_name.lower()



import cv2
import numpy as np
import os
import time

video_folder = "videos"

video_files = os.listdir(video_folder)

for video in video_files:

    print("Processing:", video)

    cap = cv2.VideoCapture(f"{video_folder}/{video}")

    danger_start_time = None

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Speed up processing
        if frame_count % 4 != 0:
            continue

        frame = cv2.resize(frame,(640,640))

        ####################################
        # PEOPLE / FIGHT / CROWD DETECTION
        ####################################

        if is_people_video:

            results = model(frame, imgsz=640, conf=0.3, verbose=False)

            detections = []

            for r in results:
                for box in r.boxes:

                    cls = int(box.cls[0])

                    if cls == 0:

                        x1,y1,x2,y2 = map(int, box.xyxy[0])

                        w = x2 - x1
                        h = y2 - y1

                        detections.append(
                            ([x1,y1,w,h],1.0,'person')
                        )

            tracks = tracker.update_tracks(detections, frame=frame)

            boxes = []

            for track in tracks:

                if not track.is_confirmed():
                    continue

                l,t,r,b = map(int, track.to_ltrb())

                boxes.append((l,t,r,b))

            ################################
            # CRASH / FIGHT DETECTION
            ################################

            danger_indices = set()

            for i in range(len(boxes)):
                for j in range(i+1, len(boxes)):

                    x1,y1,x2,y2 = boxes[i]
                    a1,b1,a2,b2 = boxes[j]

                    center1 = ((x1+x2)//2,(y1+y2)//2)
                    center2 = ((a1+a2)//2,(b1+b2)//2)

                    dist = np.linalg.norm(
                        np.array(center1) -
                        np.array(center2)
                    )

                    overlap = (
                        x1 < a2 and x2 > a1 and
                        y1 < b2 and y2 > b1
                    )

                    if dist < 60 or overlap:

                        danger_indices.add(i)
                        danger_indices.add(j)

            safe_count = 0
            danger_count = 0

            for idx,(x1,y1,x2,y2) in enumerate(boxes):

                if idx in danger_indices:

                    danger_count += 1

                    cv2.rectangle(
                        frame,
                        (x1,y1),
                        (x2,y2),
                        (0,0,255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "DANGER",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,255),
                        2
                    )

                else:

                    safe_count += 1

                    cv2.rectangle(
                        frame,
                        (x1,y1),
                        (x2,y2),
                        (0,255,0),
                        2
                    )

                    cv2.putText(
                        frame,
                        "SAFE",
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,255,0),
                        2
                    )

            ################################
            # DISPLAY COUNTS
            ################################

            people_count = len(boxes)

            cv2.putText(
                frame,
                f"People Count: {people_count}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2
            )

            cv2.putText(
                frame,
                f"Safe: {safe_count}",
                (20,70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,255,0),
                2
            )

            cv2.putText(
                frame,
                f"Danger: {danger_count}",
                (20,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0,0,255),
                2
            )

        ####################################
        # FIRE DETECTION
        ####################################

        elif is_fire_video:

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower1 = np.array([0,80,150])
            upper1 = np.array([20,255,255])

            lower2 = np.array([160,80,150])
            upper2 = np.array([180,255,255])

            mask = (
                cv2.inRange(hsv,lower1,upper1) +
                cv2.inRange(hsv,lower2,upper2)
            )

            contours,_ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:

                if cv2.contourArea(cnt) > 200:

                    x,y,w,h = cv2.boundingRect(cnt)

                    cv2.rectangle(
                        frame,
                        (x,y),
                        (x+w,y+h),
                        (0,0,255),
                        3
                    )

                    cv2.putText(
                        frame,
                        "DANGER: FIRE",
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,0,255),
                        2
                    )

        ####################################

        out.write(frame)

        cv2.imshow("AI Surveillance System", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()

    if 'frame' not in locals() or frame is None:
        continue

    frame = cv2.resize(frame,(640,480))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_fire1 = np.array([0, 80, 150])
    upper_fire1 = np.array([20, 255, 255])

    lower_fire2 = np.array([160, 80, 150])
    upper_fire2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_fire1, upper_fire1)
    mask2 = cv2.inRange(hsv, lower_fire2, upper_fire2)

    fire_mask = cv2.bitwise_or(mask1, mask2)

        # Remove noise
    kernel = np.ones((5,5),np.uint8)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel)
    fire_mask = cv2.dilate(fire_mask, kernel, iterations=2)

    contours,_ = cv2.findContours(fire_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_detected = False

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area > 150:

                fire_detected = True

                x,y,w,h = cv2.boundingRect(cnt)

                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)

                cv2.putText(frame,
                            "DANGER: FIRE",
                            (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0,0,255),
                            2)

        if fire_detected:

            if danger_start_time is None:
                danger_start_time = time.time()

            danger_time = int(time.time() - danger_start_time)

            cv2.putText(frame,
                        f"Danger Time: {danger_time}s",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0,0,255),
                        2)

        cv2.imshow("Fire Detection", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()

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

