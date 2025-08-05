import numpy as np
import cv2
from picamera2 import Picamera2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Display settings
MARGIN = -1
ROW_SIZE = 10
FONT_SIZE = 1
FONT_THICKNESS = 1
rect_color = (255, 0, 255)
TEXT_COLOR = (255, 0, 0)

# Initialize PiCamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load TFLite model
base_options = python.BaseOptions(model_asset_path="apple1.tflite")
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5
)
detector = vision.ObjectDetector.create_from_options(options)

# Main loop
while True:
    frame = picam2.capture_array()
    frame = cv2.flip(frame, -1)

    # Convert frame to MediaPipe image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Perform detection
    detection_result = detector.detect(mp_image)

    # Draw results
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x, y = int(bbox.origin_x), int(bbox.origin_y)
        w, h = int(bbox.width), int(bbox.height)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 3)

        # Draw label
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        label = f"{category_name} ({probability})"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    # Show the output
    cv2.imshow("PiCam2 Object Detection", frame)

    # Exit on ESC key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
