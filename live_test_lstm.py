import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from collections import deque
import time
import os

# ============================
# Load ALL models
# ============================

MODEL_DIR = "./trained_models"

model_paths = {
    "HELLO": {
        "model": f"{MODEL_DIR}/HELLO_vs_NOGESTURE_model/hello_vs_no_gesture_lstm_model.h5",
        "label": f"{MODEL_DIR}/HELLO_vs_NOGESTURE_model/hello_vs_no_gesture_label_encoder.pickle"
    },
    "J": {
        "model": f"{MODEL_DIR}/J_vs_NOGESTURE_model/j_vs_no_gesture_lstm_model.h5",
        "label": f"{MODEL_DIR}/J_vs_NOGESTURE_model/j_vs_no_gesture_label_encoder.pickle"
    },
    "YES": {
        "model": f"{MODEL_DIR}/YES_vs_NOGESTURE_model/yes_vs_no_gesture_lstm_model.h5",
        "label": f"{MODEL_DIR}/YES_vs_NOGESTURE_model/yes_vs_no_gesture_label_encoder.pickle"
    },
    "Z": {
        "model": f"{MODEL_DIR}/Z_vs_NOGESTURE_model/z_vs_no_gesture_lstm_model.h5",
        "label": f"{MODEL_DIR}/Z_vs_NOGESTURE_model/z_vs_no_gesture_label_encoder.pickle"
    }
}

models = {}

print("Loading models...")

for key, paths in model_paths.items():
    try:
        mdl = load_model(paths["model"])
        with open(paths["label"], "rb") as f:
            lbl = pickle.load(f)

        models[key] = {
            "model": mdl,
            "label_encoder": lbl,
            "classes": list(lbl.classes_)
        }

        print(f"✅ Loaded {key} model with classes {list(lbl.classes_)}")

    except Exception as e:
        print(f"❌ Failed to load {key}: {e}")

# ============================
# Mediapipe setup
# ============================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

# ============================
# Parameters
# ============================

SEQ_LEN = 50

BASE_MOTION_NOISE = 0.0025
motion_threshold = BASE_MOTION_NOISE * 0.9

NO_MOTION_REQUIRED = 10
COOLDOWN_FRAMES = 10
cooldown_counter = 0

conf_history = deque(maxlen=50)

sequence = []
prev_landmarks = None
no_motion_count = 0

cap = cv2.VideoCapture(0)

print("Starting in:")
for i in range(3, 0, -1):
    print(i)
    time.sleep(1)
print("🎬 Multi-model live detection started!")

stable_prediction = "Waiting for gesture..."

# ============================
# MAIN LOOP
# ============================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    display_pred = stable_prediction

    if cooldown_counter > 0:
        cooldown_counter -= 1

    if results.multi_hand_landmarks:

        # GET LANDMARKS
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        wrist = landmarks[0]
        landmarks -= wrist
        lm_flat = landmarks[:, :2].flatten()

        # COMPUTE MOTION
        if prev_landmarks is not None:
            motion = np.mean(np.abs(lm_flat - prev_landmarks))
        else:
            motion = 999

        # Auto adjust threshold first seconds
        if time.time() < 6:
            BASE_MOTION_NOISE = (BASE_MOTION_NOISE * 0.9) + (motion * 0.1)
            motion_threshold = BASE_MOTION_NOISE * 1.0

        prev_landmarks = lm_flat

        # If cooldown active → skip
        if cooldown_counter > 0:
            conf_history.append(0)
            cv2.putText(frame, "Cooldown...", (20, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Realtime Gesture", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # Detect motion
        if motion > motion_threshold:
            sequence.append(lm_flat)
            no_motion_count = 0
        else:
            no_motion_count += 1

        # If gesture ended
        if no_motion_count >= NO_MOTION_REQUIRED and len(sequence) > 20:
            seq = np.array(sequence)

            if len(seq) < SEQ_LEN:
                pad = np.zeros((SEQ_LEN - len(seq), 42))
                seq = np.vstack([seq, pad])
            else:
                seq = seq[-SEQ_LEN:]

            seq = (seq - np.mean(seq)) / (np.std(seq) + 1e-6)
            X = np.expand_dims(seq, axis=0)

            best_pred = "No gesture"
            best_conf = 0

            # ============================
            # RUN PREDICTION ON ALL MODELS
            # ============================

            for key, data in models.items():
                mdl = data["model"]
                lbl = data["label_encoder"]

                pred = mdl.predict(X, verbose=0)
                idx = np.argmax(pred)
                conf = float(np.max(pred))
                label = lbl.classes_[idx]

                # Pick highest confidence among all models
                if conf > best_conf:
                    best_conf = conf
                    best_pred = f"{label} ({conf:.2f})"

            conf_history.append(best_conf)

            if best_conf < 0.9:
                stable_prediction = "No gesture"
            else:
                stable_prediction = best_pred

            cooldown_counter = COOLDOWN_FRAMES
            sequence = []
            prev_landmarks = None

        else:
            conf_history.append(0)

    else:
        prev_landmarks = None
        conf_history.append(0)

    # ============================
    # Confidence graph
    # ============================

    graph_y = 450
    graph_height = 120
    graph_width = 300
    x_start = 20

    cv2.rectangle(frame, (x_start, graph_y),
                  (x_start + graph_width, graph_y - graph_height),
                  (50, 50, 50), 2)

    pts = []
    for i, c in enumerate(conf_history):
        x = x_start + int((i / len(conf_history)) * graph_width)
        y = graph_y - int(c * graph_height)
        pts.append((x, y))

    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), 2)

    # ============================
    # Show prediction
    # ============================

    cv2.putText(frame, display_pred, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    cv2.imshow("Realtime Gesture", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
