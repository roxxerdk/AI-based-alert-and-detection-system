import cv2
from victim_detection import detect_victims
from gesture_detection import detect_gesture
from water_segmentation import detect_water

input_path = "videos/rescue.mp4"


def process_frame(frame):

    frame = cv2.resize(frame,(640,480))

    victims = detect_victims(frame)
    water_mask = detect_water(frame)

    safe = 0
    danger = 0

    for (cx,cy) in victims:

        distress = detect_gesture(frame)

        # check if victim is standing in water
        in_water = water_mask[cy, cx] > 0

        if distress or in_water:

            color = (0,0,255)
            label = "DANGER"
            danger += 1

        else:

            color = (0,255,0)
            label = "SAFE"
            safe += 1

        cv2.circle(frame,(cx,cy),6,color,-1)

        cv2.putText(frame,label,(cx-20,cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    cv2.putText(frame,f"People: {len(victims)}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    cv2.putText(frame,f"Safe: {safe}",(20,80),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.putText(frame,f"Danger: {danger}",(20,120),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    return frame


def is_image(file):
    return file.lower().endswith((".jpg",".jpeg",".png",".bmp"))


def is_video(file):
    return file.lower().endswith((".mp4",".avi",".mov",".mkv"))


# IMAGE MODE
if is_image(input_path):

    frame = cv2.imread(input_path)

    if frame is None:
        print("Cannot open image")
        exit()

    frame = process_frame(frame)

    cv2.imshow("Flood Rescue Monitoring System", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# VIDEO MODE
elif is_video(input_path):

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Cannot open video")
        exit()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = process_frame(frame)

        cv2.imshow("Flood Rescue Monitoring System", frame)

        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


else:
    print("Unsupported file format")