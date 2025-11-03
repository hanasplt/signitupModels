# collect_videos_lstm.py
import cv2
import os
import time

DATA_DIR = './data_videos'
os.makedirs(DATA_DIR, exist_ok=True)

GESTURE_CLASSES = ['J', 'Z']
VIDEOS_PER_CLASS = 15
VIDEO_DURATION = 4  # seconds per video
FPS = 30

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot access camera.")
    exit()

print("Camera initialized. Press 's' to start collecting or 'q' to quit.")

# Camera warm-up
for _ in range(10):
    cap.read()

# Wait for user
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "Press 'S' to start or 'Q' to quit", (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Camera Feed", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        break
    elif key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Start collecting
for gesture in GESTURE_CLASSES:
    gesture_dir = os.path.join(DATA_DIR, gesture)
    os.makedirs(gesture_dir, exist_ok=True)
    print(f"\nCollecting videos for gesture '{gesture}'")

    for vid_num in range(VIDEOS_PER_CLASS):
        print(f"\nPreparing to record video {vid_num+1}/{VIDEOS_PER_CLASS}")

        # Countdown
        for countdown in range(3, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.putText(frame, f"{gesture}: Starting in {countdown}", (100, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
            cv2.imshow("Camera Feed", frame)
            cv2.waitKey(1000)

        # Setup video writer
        filename = os.path.join(gesture_dir, f"{gesture}_{vid_num}.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        out = cv2.VideoWriter(filename, fourcc, FPS, (frame_width, frame_height))

        print("🎬 Recording started!")
        start_time = time.time()

        while (time.time() - start_time) < VIDEO_DURATION:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            out.write(frame)
            cv2.putText(frame, f"Recording: {gesture} {vid_num+1}/{VIDEOS_PER_CLASS}",
                        (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            cv2.imshow("Camera Feed", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        out.release()
        print(f"✅ Saved video: {filename}")
        time.sleep(1)

cap.release()
cv2.destroyAllWindows()
print("\n🎉 All video captures complete!")
