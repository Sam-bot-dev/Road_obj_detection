from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
video_path = "/content/road.mp4"
cap = cv2.VideoCapture(video_path)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("detected.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"),
                      fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=[frame], conf=0.25, verbose=False)
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

cap.release()
out.release()

# Play inside Colab
from IPython.display import Video
Video("detected.mp4", embed=True)
