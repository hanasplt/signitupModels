import cv2
import mediapipe as mp
import numpy as np
import pickle
import onnxruntime as ort
from collections import deque
import time

# ============================
# Load ONNX Model + Labels
# ============================

MODEL_PATH = "web_demo/gesture_lstm.onnx"
LABEL_ENCODER_PATH = "web_demo/label_encoder.pickle"

print("Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

CLASSES = list(label_encoder.classes_)
print("Loaded classes:", CLASSES)

# ============================
# Mediapipe Setup
# ============================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

# ============================
# Parameters
# ============================

SEQ_LEN = 50
BASE_MOTION_NOISE = 0.0025
motion_threshold = BASE_MOTION_NOISE

NO_MOTION_REQUIRED = 10
COOLDOWN_FRAMES = 10

sequence = []
prev_landmarks = None
no_motion_count = 0
cooldown_counter = 0

conf_history = deque(maxlen=50)

cap = cv2.VideoCapture(0)

print("Starting in:")
for i in range(3, 0, -1):
    print(i)
    time.sleep(1)

print("🎬 ONNX Real-time Detection Started!")

stable_prediction = "Waiting for gesture..."

# ============================
# Main Loop
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

        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract normalized landmarks
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        wrist = landmarks[0]
        landmarks -= wrist  # wrist-relative
        lm_flat = landmarks[:, :2].flatten()  # 21 * 2 = 42 dims

        # Motion computation
        if prev_landmarks is not None:
            motion = np.mean(np.abs(lm_flat - prev_landmarks))
        else:
            motion = 999

        # Auto adjust noise baseline (first 6 seconds)
        if time.time() < 6:
            BASE_MOTION_NOISE = (BASE_MOTION_NOISE * 0.9) + (motion * 0.1)
            motion_threshold = BASE_MOTION_NOISE * 1.0

        prev_landmarks = lm_flat

        # Skip if cooldown active
        if cooldown_counter > 0:
            conf_history.append(0)
            cv2.putText(frame, "Cooldown...", (20, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Realtime Gesture (ONNX)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # Motion detection
        if motion > motion_threshold:
            sequence.append(lm_flat)
            no_motion_count = 0
        else:
            no_motion_count += 1

        # Gesture ended → Run model
        if no_motion_count >= NO_MOTION_REQUIRED and len(sequence) > 20:

            seq = np.array(sequence)

            # Pad/truncate sequence
            if len(seq) < SEQ_LEN:
                pad = np.zeros((SEQ_LEN - len(seq), 42))
                seq = np.vstack([seq, pad])
            else:
                seq = seq[-SEQ_LEN:]

            # Normalize
            seq = (seq - np.mean(seq)) / (np.std(seq) + 1e-6)
            X = np.expand_dims(seq.astype(np.float32), axis=0)  # (1, 50, 42)

            # Run ONNX inference
            pred = session.run(None, {input_name: X})[0]
            idx = np.argmax(pred)
            conf = float(np.max(pred))
            label = CLASSES[idx]

            conf_history.append(conf)

            if conf >= 0.90:
                stable_prediction = f"{label} ({conf:.2f})"
            else:
                stable_prediction = "No gesture"

            cooldown_counter = COOLDOWN_FRAMES
            sequence = []
            prev_landmarks = None

        else:
            conf_history.append(0)

    else:
        prev_landmarks = None
        conf_history.append(0)

    # ============================
    # Confidence Graph
    # ============================

    graph_y = 450
    graph_h = 120
    graph_w = 300
    x0 = 20

    cv2.rectangle(frame, (x0, graph_y), (x0 + graph_w, graph_y - graph_h),
                  (50, 50, 50), 2)

    pts = []
    for i, c in enumerate(conf_history):
        x = x0 + int((i / len(conf_history)) * graph_w)
        y = graph_y - int(c * graph_h)
        pts.append((x, y))

    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), 2)

    # ============================
    # Display Prediction
    # ============================

    cv2.putText(frame, display_pred, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    cv2.imshow("Realtime Gesture (ONNX)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
