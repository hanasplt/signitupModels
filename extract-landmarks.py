# extract_landmarks_sequence_v5_all_VISUAL.py
import os
os.environ["MEDIAPIPE_DISABLE_TF_IMPORT"] = "1"
import cv2
import mediapipe as mp
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------¬
#  SHOW VIDEO + LANDMARKS  ←---
# -----------------------------¬
DEBUG_VISUALIZE = True      # ← flip to True
INCLUDE_IDLE    = False     # skip totally still frames while watching
# ---------------------------------------------------------
#  Everything else stays identical except an extra status
# ---------------------------------------------------------
DATA_DIR       = './data_videos'
OUTPUT_DIR     = './processed_data'
OUTPUT_FILE    = os.path.join(OUTPUT_DIR, 'dynamic_gestures_data_VISUAL.p')
SEQUENCE_LENGTH = 50
MOTION_THRESHOLD = 0.001
PLOT_SAMPLE = True

os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

data, labels, class_names = [], [], []
print(f"🎥 Scanning all gesture folders in: {DATA_DIR}\n")

for gesture_folder in sorted(os.listdir(DATA_DIR)):
    gesture_path = os.path.join(DATA_DIR, gesture_folder)
    if not os.path.isdir(gesture_path):
        continue

    class_names.append(gesture_folder)
    print(f"📁 Processing gesture folder: {gesture_folder}")

    for video_file in os.listdir(gesture_path):
        if not video_file.lower().endswith(('.avi', '.mp4', '.mov')):
            continue

        video_path = os.path.join(gesture_path, video_file)
        cap = cv2.VideoCapture(video_path)

        sequence, prev_landmarks, frame_index, skipped_frames = [], None, 0, 0
        print(f"   ▶️ Processing video: {video_file}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y])

                    # -------- DRAW ---------
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style())

                # motion filter (same as before)
                if prev_landmarks is not None:
                    motion = np.mean(np.abs(np.array(landmarks) - np.array(prev_landmarks)))
                    if motion < MOTION_THRESHOLD and not INCLUDE_IDLE:
                        skipped_frames += 1
                        continue
                prev_landmarks = landmarks
                sequence.append(landmarks)
            else:
                sequence.append([0] * 42)

            # --------------- SHOW WINDOW ---------------
            preview = cv2.flip(frame, 1)  # mirror feels natural
            cv2.putText(preview, f"{gesture_folder}  frame:{frame_index}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(preview, "[ESC] skip video", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.imshow("Landmark Extraction Preview", preview)
            # ------------------------------------------

            frame_index += 1
            if cv2.waitKey(1) & 0xFF == 27:  # ESC skips rest of this video
                break

        cap.release()

        # pad / trim
        sequence = np.array(sequence[:SEQUENCE_LENGTH])
        if len(sequence) < SEQUENCE_LENGTH:
            pad = np.zeros((SEQUENCE_LENGTH - len(sequence), 42))
            sequence = np.vstack((sequence, pad))

        data.append(sequence)
        labels.append(gesture_folder)
        print(f"   ✅ {gesture_folder}: {len(sequence)} frames | Skipped: {skipped_frames}")

cv2.destroyAllWindows()

# save exactly like before
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels, 'classes': class_names}, f)

print(f"\n🎯 All gestures extracted with visual feedback!")
print(f"📦 Saved dataset to: {OUTPUT_FILE}")
print(f"🧩 Classes: {class_names}")

# optional plot
if PLOT_SAMPLE and len(data) > 0:
    seq = np.array(data[0])
    plt.figure(figsize=(8, 4))
    plt.plot(seq[:, 0], label='x of landmark 0')
    plt.title(f"Motion of first landmark for '{labels[0]}'")
    plt.xlabel("Frame index")
    plt.ylabel("Normalized position")
    plt.legend()