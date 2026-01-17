# ==========================================================
# STABLE REAL-TIME HAND GESTURE INFERENCE (ONNX)
# ==========================================================

import cv2
import mediapipe as mp
import numpy as np
import pickle
import onnxruntime as ort
from collections import deque
import time

# ==========================================================
# CONFIG
# ==========================================================

MODEL_PATH = "web_demo/gesture_lstm.onnx"
LABEL_ENCODER_PATH = "web_demo/label_encoder.pickle"

SEQ_LEN = 50
VECTOR_LEN = 42

CONF_THRESHOLD = 0.65
NO_MOTION_REQUIRED = 10
COOLDOWN_FRAMES = 12
BASE_MOTION_NOISE = 0.0025
SMOOTHING_ALPHA = 0.75

# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)

CLASSES = list(le.classes_)
print("Classes:", CLASSES)

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum(axis=1, keepdims=True)

# ==========================================================
# MEDIAPIPE
# ==========================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ==========================================================
# WEBCAM
# ==========================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Camera error")

print("Starting...")
time.sleep(2)

# ==========================================================
# STATE
# ==========================================================

sequence = []
prev_lm = None
smooth_lm = None
no_motion = 0
cooldown = 0

conf_history = deque(maxlen=60)

prediction = "Waiting..."
confidence = 0.0

# ==========================================================
# MAIN LOOP (INFERENCE)
# ==========================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    motion = 0.0

    if cooldown > 0:
        cooldown -= 1

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        lm = np.array([[p.x, p.y] for p in hand.landmark])
        lm -= lm[0]                    # wrist-relative
        lm = lm.flatten()

        # ---- EMA smoothing ----
        if smooth_lm is None:
            smooth_lm = lm
        else:
            smooth_lm = SMOOTHING_ALPHA * smooth_lm + (1 - SMOOTHING_ALPHA) * lm

        # ---- motion detection ----
        if prev_lm is not None:
            motion = np.mean(np.abs(smooth_lm - prev_lm))
        else:
            motion = 999

        prev_lm = smooth_lm.copy()

        cv2.putText(frame, f"Motion: {motion:.5f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # ---- motion logic ----
        if cooldown == 0:
            if motion > BASE_MOTION_NOISE:
                sequence.append(smooth_lm)
                no_motion = 0
            else:
                no_motion += 1

        # ---- inference trigger ----
        if no_motion >= NO_MOTION_REQUIRED and len(sequence) >= 20:
            seq = np.array(sequence)

            if len(seq) < SEQ_LEN:
                seq = np.vstack([seq, np.zeros((SEQ_LEN-len(seq), VECTOR_LEN))])
            else:
                seq = seq[-SEQ_LEN:]

            # 🔥 MATCH TRAINING NORMALIZATION 🔥
            seq = (seq - seq.min()) / (seq.max() - seq.min() + 1e-6)

            X = np.expand_dims(seq.astype(np.float32), axis=0)

            logits = session.run(None, {input_name: X})[0]
            probs = softmax(logits)

            idx = np.argmax(probs)
            confidence = float(probs[0][idx])

            prediction = CLASSES[idx] if confidence >= CONF_THRESHOLD else "Uncertain"

            conf_history.append(confidence)

            # reset
            sequence.clear()
            prev_lm = None
            smooth_lm = None
            no_motion = 0
            cooldown = COOLDOWN_FRAMES

        else:
            conf_history.append(0)

    else:
        prev_lm = None
        smooth_lm = None
        conf_history.append(0)

    # ======================================================
    # VISUALS
    # ======================================================

    cv2.putText(frame, f"Prediction: {prediction}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.putText(frame, f"Confidence: {confidence:.2f}", (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    # confidence graph
    gy, gh, gw = 460, 120, 300
    x0 = 20
    cv2.rectangle(frame, (x0, gy), (x0+gw, gy-gh), (50,50,50), 2)

    pts = [(x0 + int(i/gw*gw), gy - int(c*gh))
           for i, c in enumerate(conf_history)]

    for i in range(1, len(pts)):
        cv2.line(frame, pts[i-1], pts[i], (0,255,0), 2)

    cv2.imshow("Stable ONNX Gesture Inference", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
