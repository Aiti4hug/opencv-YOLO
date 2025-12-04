from ultralytics import YOLO
import os
import cv2
import datetime

model = YOLO("yolov8n.pt")

video_path = 'VIDEOS'
os.makedirs(video_path, exist_ok=True)

cop = 'MEdIA'
os.makedirs(cop, exist_ok=True)

cam = cv2.VideoCapture('gg/airplane_video1.mp4')


if not cam:
    print('camera not found')
    exit()

counter = 0

frame_w = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = float(cam.get(cv2.CAP_PROP_FPS))


if frame_fps == 0:
    frame_fps = 30.0
video_date = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
type_video = cv2.VideoWriter_fourcc(*'mp4v')

while True:
    ret, frame = cam.read()
    if not ret:
        print('Frame not found')
        break

    results = model(frame, classes=[2, 4, 15, 16])

    tactics_frame = results[0].plot()

    date = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    cv2.putText(tactics_frame, date, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('v'):
        video_name = f"{video_path}/video_{video_date}.mp4"
        video = cv2.VideoWriter(video_name, type_video, frame_fps,
                                (frame_w, frame_h))
        video.write(tactics_frame)
    elif key == ord('t'):
        cv2.imshow('Video', tactics_frame)
    elif key == ord('s'):
        date_type = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        image_name = f'{cop}/image_{date_type}.png'
        cv2.imwrite(image_name, tactics_frame)
        counter +=1
        print(f'image number {counter}')

        cv2.putText(tactics_frame, 'Save', (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.imshow('S for take photo', tactics_frame)
        cv2.waitKey(300)
    else:
        cv2.imshow('Video', tactics_frame)

cam.release()
cv2.destroyAllWindows()