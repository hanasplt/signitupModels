# ==========================================================
# STABLE REAL-TIME HAND GESTURE INFERENCE (ONNX) — FIXED RANK
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
MODEL_PATH = "web_demo/FAMILY/gesture_lstm_FAMILY.onnx"
LABEL_ENCODER_PATH = "web_demo/FAMILY/label_encoder_FAMILY.pickle"

SEQ_LEN = 50
VECTOR_LEN = 89 

CONF_THRESHOLD = 0.65
NO_MOTION_REQUIRED = 8 
COOLDOWN_FRAMES = 15
BASE_MOTION_NOISE = 0.003
SMOOTHING_ALPHA = 0.6

# ==========================================================
# LOAD MODEL + NORMALIZATION
# ==========================================================
print("Initializing Inference Engine...")
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

with open(LABEL_ENCODER_PATH, "rb") as f:
    le = pickle.load(f)
CLASSES = list(le.classes_)

# ✅ FIXED: Flatten normalization arrays to prevent Rank/Dimension errors
norm_mean = np.load("web_demo/FAMILY/norm_mean_FAMILY.npy").flatten().astype(np.float32)
norm_std  = np.load("web_demo/FAMILY/norm_std_FAMILY.npy").flatten().astype(np.float32)

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
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7
)

# ==========================================================
# STATE & UTILS
# ==========================================================
cap = cv2.VideoCapture(0)
sequence = []
prev_lm = None
smooth_lm = None
no_motion = 0
cooldown = 0
prediction = "Waiting..."
confidence = 0.0

def compute_finger_states(lm_flat):
    """Calculates semantic states (Open/Closed) based on Y-coordinates."""
    p = lm_flat.reshape(21, 2)
    return np.array([
        p[4,1]  < p[3,1],  # Thumb
        p[8,1]  < p[6,1],  # Index
        p[12,1] < p[10,1], # Middle
        p[16,1] < p[14,1], # Ring
        p[20,1] < p[18,1]  # Pinky
    ], dtype=np.float32)

# ==========================================================
# MAIN LOOP
# ==========================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    motion = 0.0
    if cooldown > 0: cooldown -= 1

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        # 1. Coordinate Extraction
        # Note: We flatten to 42 features (21 joints * 2 coords)
        lm = np.array([[p.x, p.y] for p in hand.landmark], dtype=np.float32)
        lm_flat = lm.flatten()

        # 2. EMA Smoothing
        if smooth_lm is None: smooth_lm = lm_flat
        else: smooth_lm = SMOOTHING_ALPHA * smooth_lm + (1 - SMOOTHING_ALPHA) * lm_flat

        # 3. Features: Velocity & Finger States
        if prev_lm is None:
            vel = np.zeros(42, dtype=np.float32)
        else:
            vel = smooth_lm - prev_lm
            motion = np.mean(np.abs(vel))

        fstate = compute_finger_states(smooth_lm)
        
        # Combine into the 89-vector used in training
        semantic_frame = np.concatenate([smooth_lm, vel, fstate])
        prev_lm = smooth_lm.copy()

        # 4. Recording Logic
        if cooldown == 0:
            if motion > BASE_MOTION_NOISE:
                sequence.append(semantic_frame)
                no_motion = 0
                cv2.putText(frame, "RECORDING...", (20, 450), 1, 1, (0,0,255), 2)
            else:
                no_motion += 1

        # 5. Trigger Inference
        if no_motion >= NO_MOTION_REQUIRED and len(sequence) >= 15:
            seq_arr = np.array(sequence, dtype=np.float32)
            
            # ✅ FIXED: Padding by repeating the last frame context
            if len(seq_arr) < SEQ_LEN:
                padding = np.tile(seq_arr[-1], (SEQ_LEN - len(seq_arr), 1))
                seq_arr = np.vstack([seq_arr, padding])
            else:
                seq_arr = seq_arr[-SEQ_LEN:]

            # ✅ FIXED: Normalization and Rank Handling
            # Subtract 1D norm arrays from 2D sequence (Broadcasting)
            seq_arr = (seq_arr - norm_mean) / (norm_std + 1e-6)
            
            # Expand to 3D: (1, 50, 89)
            input_data = np.expand_dims(seq_arr, axis=0).astype(np.float32)

            # Run ONNX Session
            logits = session.run(None, {input_name: input_data})[0]
            probs = softmax(logits)
            
            idx = np.argmax(probs)
            confidence = float(probs[0][idx])

            if confidence >= CONF_THRESHOLD:
                prediction = CLASSES[idx]
            else:
                prediction = "Uncertain"

            # Reset state for next gesture
            sequence = []
            cooldown = COOLDOWN_FRAMES
            no_motion = 0

    # UI Display
    cv2.rectangle(frame, (0,0), (350, 120), (0,0,0), -1)
    cv2.putText(frame, f"PRED: {prediction}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f"CONF: {confidence:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1)
    
    cv2.imshow("FSL Dynamic Inference", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()