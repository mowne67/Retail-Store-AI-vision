import shutil
import tempfile
import streamlit as st
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
model = YOLO('yolov8n.pt')

video_bytes = st.file_uploader("Upload a video", type=["mp4"])

if video_bytes:
    with open("input_video.mp4", "wb") as f:
        f.write(video_bytes.read())

# with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
#   temp_filename = temp_file.name
#   temp_file.close()
#   shutil.move('input_video.mp4', temp_filename)

video_path = 'input_video.mp4'
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
result = cv2.VideoWriter('store_track.mp4',
                         cv2.VideoWriter_fourcc(*'H264'),
                         10, size)
track_history = defaultdict(lambda: [])

with st.spinner("Processing"):
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(source=frame, persist=True)

            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            annotated_frame = results[0].plot()
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y+h/2)))  # x, y center point
                # if len(track) > 30:  # retain 90 tracks for 90 frames
                #     track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 230, 230), thickness=5)

            #cv2.imshow("YOLOv8 Inference", annotated_frame)
            result.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    result.release()
    cv2.destroyAllWindows()

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file1:
      temp_filename1 = temp_file1.name
      temp_file1.close()

      # Move the video file to the temporary file
      # shutil.move('store_track.mp4', temp_filename1)
video_file = open('store_track.mp4', 'rb')
video_bytes = video_file.read()

st.video(video_bytes)




