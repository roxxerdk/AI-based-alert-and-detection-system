from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import time

app = Flask(__name__)

model = YOLO("yolov8n.pt")

# Global variables to store state
current_count = 0
is_overcrowded = False

def generate_frames():
    global current_count, is_overcrowded
    cap = cv2.VideoCapture("test_video.mp4") # Change to 0 for webcam

    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
            continue

        frame = cv2.resize(frame, (640, 480))
        frame_count += 1
        if frame_count % 3 != 0: # Skip frames for speed
            continue

        results = model(frame)
        
        # Reset count for this frame
        temp_count = 0
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0:
                    temp_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Dynamic color: Red if high density, Cyan if normal
                    color = (0, 0, 255) if temp_count > 50 else (255, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Update global state
        current_count = temp_count
        is_overcrowded = current_count > 50

        # Encode
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# NEW: API to send stats to the UI
@app.route('/stats')
def stats():
    return jsonify({
        'count': current_count,
        'alert': is_overcrowded
    })

if __name__ == "__main__":
    app.run(debug=True)