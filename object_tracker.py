import cv2
import sqlite3
import numpy as np
import pyttsx3
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import threading
import speech_recognition as sr
import time
import math

# Load YOLOv8 model (You can fine-tune this model for better accuracy)
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=50)

# Define target and surface objects
TARGET_OBJECTS = ["cell phone", "pen", "bottle", "toothbrush", "rubik's cube"]
SURFACE_OBJECTS = ["table", "bed", "desk", "sofa", "chair", "floor"]

# SQLite DB for storing object info
conn = sqlite3.connect("object_locations.db")
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS objects (
        id TEXT PRIMARY KEY,
        name TEXT,
        color TEXT,
        location TEXT,
        timestamp REAL
    )
''')
conn.commit()

# Voice engine init
engine = pyttsx3.init()

# Voice command listener thread
def listen_for_commands():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    conn_thread = sqlite3.connect("object_locations.db")
    cursor_thread = conn_thread.cursor()

    while True:
        try:
            with mic as source:
                print("🎤 Listening for command...")
                audio = recognizer.listen(source, phrase_time_limit=5)
            command = recognizer.recognize_google(audio).lower()
            print("Command:", command)

            for obj in TARGET_OBJECTS:
                if obj in command:
                    cursor_thread.execute("SELECT location FROM objects WHERE name = ?", (obj,))
                    rows = cursor_thread.fetchall()
                    if rows:
                        locations = list(set([r[0] for r in rows]))
                        response = f"{obj} is located at " + ", ".join(locations)
                    else:
                        response = f"I could not detect any {obj}."
                    print("📢", response)
                    engine.say(response)
                    engine.runAndWait()
        except Exception as e:
            print("Voice command error:", e)

# Start voice thread
command_thread = threading.Thread(target=listen_for_commands, daemon=True)
command_thread.start()

# Video feed
cap = cv2.VideoCapture(0)

# Helper functions
def get_center(x1, y1, x2, y2):
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def calculate_distance(c1, c2):
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

# Main processing loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    surface_locations = []
    detections = []

    # Detect objects
    results = model(frame)

    for result in results:
        boxes = result.boxes
        names = result.names

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            label = names[class_id]

            if conf < 0.5:
                continue

            center = get_center(x1, y1, x2, y2)

            if label in SURFACE_OBJECTS:
                surface_locations.append((label, center))

            if label in TARGET_OBJECTS:
                detections.append(([x1, y1, x2, y2], conf, label))

    # Track objects
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = str(track.track_id)
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        center = get_center(x1, y1, x2, y2)
        det_class = track.get_det_class() or "unknown"

        # Determine object surface proximity
        location = "in unknown area"
        min_dist = float("inf")
        for surface_label, surface_center in surface_locations:
            dist = calculate_distance(center, surface_center)
            if dist < min_dist and dist < 200:
                location = f"on {surface_label}"
                min_dist = dist

        # Get object color
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            color = np.mean(crop, axis=(0, 1)).astype(int).tolist()
        else:
            color = [0, 0, 0]

        # Update database
        cursor.execute('''
            INSERT OR REPLACE INTO objects (id, name, color, location, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (track_id, det_class, str(color), location, time.time()))
        conn.commit()

        # Draw box
        label_text = f"{det_class} ({location})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"✅ {det_class} detected at {location} | Color: {color}")

    # Show the result
    cv2.imshow("📷 Object Tracker", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
conn.close()
cv2.destroyAllWindows()
