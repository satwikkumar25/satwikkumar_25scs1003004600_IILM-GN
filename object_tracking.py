import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("yolo11n.pt")

# -----------------------------
# Create Deep SORT tracker
# -----------------------------
tracker = DeepSort(
    max_age=30,
    n_init=2,
    max_cosine_distance=0.4
)

# -----------------------------
# Open camera
# -----------------------------
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera started.")
print("Press Q to quit.")

frame_count = 0
tracks = []

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    # --------------------------------
    # Run YOLO every 2nd frame
    # --------------------------------
    if frame_count % 2 == 0:

        results = model(
            frame,
            imgsz=640,
            conf=0.30,
            verbose=False
        )

        detections = []

        for result in results:

            for box in result.boxes:

                confidence = float(box.conf[0])

                if confidence < 0.30:
                    continue

                # Coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Class
                class_id = int(box.cls[0])
                class_name = model.names[class_id]

                # Width and height
                width = x2 - x1
                height = y2 - y1

                # Deep SORT detection format
                detections.append(
                    (
                        [x1, y1, width, height],
                        confidence,
                        class_name
                    )
                )

        # --------------------------------
        # Update Deep SORT
        # --------------------------------
        tracks = tracker.update_tracks(
            detections,
            frame=frame
        )

    # --------------------------------
    # Draw tracking information
    # --------------------------------
    for track in tracks:

        if not track.is_confirmed():
            continue

        # Get tracking ID
        track_id = track.track_id

        # Get bounding box
        x1, y1, x2, y2 = track.to_ltrb()

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # Get class
        class_name = track.get_det_class()

        if class_name is None:
            class_name = "Object"

        # --------------------------------
        # Draw bounding box
        # --------------------------------
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # --------------------------------
        # Label with ID
        # --------------------------------
        label = f"{class_name} | ID: {track_id}"

        # Label background
        (text_width, text_height), _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        cv2.rectangle(
            frame,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 10, y1),
            (0, 255, 0),
            -1
        )

        # Text
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    # --------------------------------
    # Display
    # --------------------------------
    cv2.imshow(
        "Object Detection + Deep SORT Tracking",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()

print("Program stopped.")