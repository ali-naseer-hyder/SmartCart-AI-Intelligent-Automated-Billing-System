from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import time

app = Flask(__name__)
model = YOLO('yolov8n.pt')

model = YOLO('weights/yolov8n.pt')# Ensure this file is in your SmartCart folder

# AI SmartCart Price Database

PRICES = {
    # Produce
    "apple": 0.50, 
    "banana": 0.30, 
    "orange": 0.60,
    "broccoli": 1.20,
    "carrot": 0.40,
    # Drinks & Containers
    "bottle": 1.00, 
    "wine glass": 5.00,
    "cup": 1.50,
    "bowl": 2.00,
    # Snacks (Mapped to 'cake' or 'donut' in YOLO)
    "cake": 4.50,
    "donut": 1.00,
    "sandwich": 3.50,
    "pizza": 12.00,
    # Personal Items (Items someone might "drop" in a cart)
    "backpack": 25.00,
    "handbag": 40.00,
    "umbrella": 10.00,
    "tie": 15.00,
    "suitcase": 60.00,
    # Kitchenware
    "fork": 0.75,
    "knife": 0.75,
    "spoon": 0.75,
    # Tech (Optional)
    "cell phone": 599.00,
    "laptop": 999.00,
    "mouse": 20.00,
    "remote": 15.00,
    # Default/Safety
    "person": 0.00
}
cart_data = {"total": 0.0, "items": []}
billed_ids = set()
last_billed_time = {}

def gen_frames():
    cap = cv2.VideoCapture(0) # 0 is your webcam
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Run YOLO Tracking
        results = model.track(frame, persist=True, conf=0.3, tracker="bytetrack.yaml")
        
        # Billing Logic
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            current_time = time.time()

            for obj_id, cls in zip(ids, clss):
                label = model.names[cls].lower()
                # Unique ID check + 3-second cooldown per item type
                if obj_id not in billed_ids:
                    last_time = last_billed_time.get(label, 0)
                    if (current_time - last_time) > 3.0:
                        billed_ids.add(obj_id)
                        price = PRICES.get(label, 0.0)
                        cart_data["total"] += price
                        cart_data["items"].append(f"{label.capitalize()} - ${price:.2f}")
                        last_billed_time[label] = current_time

        # Encode the frame for the website
        ret, buffer = cv2.imencode('.jpg', results[0].plot())
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_cart')
def get_cart():
    return jsonify(cart_data)

@app.route('/reset')
def reset():
    global billed_ids, cart_data, last_billed_time
    billed_ids = set()
    last_billed_time = {}
    cart_data = {"total": 0.0, "items": []}
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)