# extract_landmarks_sequence_v4.py
import os
os.environ["MEDIAPIPE_DISABLE_TF_IMPORT"] = "1"
import cv2
import mediapipe as mp
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR = './data_videos'          # Base folder containing gesture folders
GESTURE_FOLDER = 'Z'                # 👈 Folder name to process
OUTPUT_DIR = './processed_data'     # 👈 Folder where output .p file will be saved
SEQUENCE_LENGTH = 50
MOTION_THRESHOLD = 0.001
INCLUDE_IDLE = True
DEBUG_VISUALIZE = True
PLOT_SAMPLE = True

# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define output file path
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f'{GESTURE_FOLDER.lower()}_data.p')

# -----------------------------
# INITIALIZE MEDIAPIPE HANDS
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------
# EXTRACTION
# -----------------------------
data = []
labels = []

gesture_path = os.path.join(DATA_DIR, GESTURE_FOLDER)

if not os.path.isdir(gesture_path):
    raise FileNotFoundError(f"❌ Folder '{GESTURE_FOLDER}' not found in {DATA_DIR}")

print(f"🎥 Extracting hand landmarks from gesture folder: {GESTURE_FOLDER}\n")

for video_file in os.listdir(gesture_path):
    if not video_file.lower().endswith(('.avi', '.mp4', '.mov')):
        continue

    video_path = os.path.join(gesture_path, video_file)
    cap = cv2.VideoCapture(video_path)

    sequence = []
    prev_landmarks = None
    frame_index = 0
    skipped_frames = 0

    print(f"▶️ Processing: {video_file}")

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

                # 🟢 Visualize live if enabled
                if DEBUG_VISUALIZE:
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            # Compute motion difference from previous frame
            if prev_landmarks is not None:
                motion = np.mean(np.abs(np.array(landmarks) - np.array(prev_landmarks)))
                if motion < MOTION_THRESHOLD:
                    skipped_frames += 1
                    continue
            prev_landmarks = landmarks
            sequence.append(landmarks)
        else:
            # No hand detected → fill with zeros
            sequence.append([0] * 42)

        # 🟢 Show visual preview
        if DEBUG_VISUALIZE:
            preview = cv2.flip(frame, 1)
            cv2.putText(preview, f"{GESTURE_FOLDER} ({frame_index})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("Landmark Extraction Preview", preview)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to stop
                break

        frame_index += 1

    cap.release()

    # Handle padding or trimming
    sequence = np.array(sequence[:SEQUENCE_LENGTH])
    if len(sequence) < SEQUENCE_LENGTH:
        pad = np.zeros((SEQUENCE_LENGTH - len(sequence), 42))
        sequence = np.vstack((sequence, pad))

    # Append to dataset
    data.append(sequence)
    labels.append(GESTURE_FOLDER)

    print(f"✅ {GESTURE_FOLDER}: {len(sequence)} frames | Skipped: {skipped_frames}")

if DEBUG_VISUALIZE:
    cv2.destroyAllWindows()

# -----------------------------
# SAVE OUTPUT
# -----------------------------
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print(f"\n🎯 Landmarks for '{GESTURE_FOLDER}' extracted and saved to {OUTPUT_FILE}")

# -----------------------------
# OPTIONAL: PLOT SAMPLE
# -----------------------------
if PLOT_SAMPLE and len(data) > 0:
    seq = np.array(data[0])
    plt.figure(figsize=(8, 4))
    plt.plot(seq[:, 0], label='x of landmark 0')
    plt.title(f"Motion of first landmark for '{labels[0]}'")
    plt.xlabel("Frame index")
    plt.ylabel("Normalized position")
    plt.legend()
    plt.show()
