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

        frame = cv2.resize(frame,(640,480))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 🔥 Wider fire color range
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

cv2.destroyAllWindows()