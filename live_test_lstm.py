# ==========================================================
# 0.  INSTALL THESE FIRST (one-time in terminal/command line)
# ----------------------------------------------------------
#   pip install opencv-python mediapipe numpy onnxruntime
# ==========================================================

import cv2                 # webcam handling + drawing
import mediapipe as mp     # Google hand-landmarker
import numpy as np         # fast math on arrays
import pickle              # load Python objects saved to disk
import onnxruntime as ort  # run the exported ONNX neural-net
from collections import deque   # fast queue for confidence graph
import time                # small delays / timers

# ==========================================================
# 1.  LOAD THE NEURAL-NET + LABEL NAMES
# ----------------------------------------------------------
# We trained a PyTorch LSTM → exported to ONNX.
# We also saved the label-encoder (sklearn) so we can turn
# prediction numbers ("class 3") into human words ("HELLO").
# ==========================================================

MODEL_PATH = "web_demo/gesture_lstm.onnx"          # the neural net
LABEL_ENCODER_PATH = "web_demo/label_encoder.pickle"  # the label list

print("Loading ONNX model...")
session = ort.InferenceSession(MODEL_PATH)         # create ONNX runtime
input_name = session.get_inputs()[0].name          # net input node name

# open the small file that contains the list of gesture names
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

CLASSES = list(label_encoder.classes_)   # ['HELLO', 'J', 'YES', ...]
print("Loaded classes:", CLASSES)

# ==========================================================
# 2.  MEDIAPIPE SET-UP
# ----------------------------------------------------------
# MediaPipe Hands gives us 21 key-points (x,y,z) for every hand
# in the camera image. We only need 1 hand.
# ==========================================================

mp_hands = mp.solutions.hands          # shortcut
mp_drawing = mp.solutions.drawing_utils # helper to draw skeleton

hands = mp_hands.Hands(
        max_num_hands=1,               # only 1 hand please
        min_detection_confidence=0.9)  # 50 % sure there is a hand

# ==========================================================
# 3.  PARAMETERS YOU CAN TWEAK
# ----------------------------------------------------------
# SEQ_LEN              : how many hand-frames the net looks at
# BASE_MOTION_NOISE    : smallest movement we still consider "motion"
# NO_MOTION_REQUIRED   : how many *still* frames trigger the network
# COOLDOWN_FRAMES      : ignore new gestures for X frames after one
# ==========================================================

SEQ_LEN = 50
BASE_MOTION_NOISE = 0.005 #0.0025
motion_threshold = BASE_MOTION_NOISE

NO_MOTION_REQUIRED = 10
COOLDOWN_FRAMES = 10

# circular buffer for nice confidence graph
conf_history = deque(maxlen=50)

# ==========================================================
# 4.  WEBCAM INITIALISATION
# ----------------------------------------------------------
# 0 = default camera (laptop webcam). Press ESC later to quit.
# ==========================================================

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# small countdown so you can move your hand in view
print("Starting in:")
for i in range(3, 0, -1):
    print(i)
    time.sleep(1)
print("🎬 ONNX Real-time Detection Started!")

# variables we will update every frame
sequence = []          # list of 42-D vectors
prev_landmarks = None  # previous vector (to compute motion)
no_motion_count = 0    # how many frames we have been still
cooldown_counter = 0   # frames left to ignore new gestures
stable_prediction = "Waiting for gesture..."

# ==========================================================
# 5.  MAIN LOOP – runs forever until you press ESC
# ----------------------------------------------------------
# Each loop:
#   - read 1 camera frame
#   - find hand landmarks
#   - decide if hand is moving or still
#   - when still long enough → run ONNX model
#   - draw results + confidence graph
# ==========================================================

while True:
    ret, frame = cap.read()      # ret = success?  frame = image
    if not ret:                  # camera problem → quit
        break

    # mirror image so it feels like a mirror
    frame = cv2.flip(frame, 1)

    # MediaPipe needs RGB, OpenCV gives BGR
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb) # ← heavy lifting here

    # default text we will show
    display_pred = stable_prediction

    # reduce cooldown every frame
    if cooldown_counter > 0:
        cooldown_counter -= 1

    # ============== HAND FOUND ? =========================
    if results.multi_hand_landmarks:
        # pick first (only) hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # draw pretty skeleton on top of hand
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # convert 21 landmarks → numpy array (21×3)
        landmarks = np.array([[lm.x, lm.y, lm.z]
                              for lm in hand_landmarks.landmark])

        # make coordinates relative to wrist (so hand position
        # in image does not matter, only shape)
        wrist = landmarks[0]
        landmarks -= wrist

        # flatten to 42 numbers (x,y only)
        lm_flat = landmarks[:, :2].flatten()

        # compute motion compared to previous frame
        if prev_landmarks is not None:
            motion = np.mean(np.abs(lm_flat - prev_landmarks))
        else:
            motion = 999  # first frame → big number

        # auto-tune noise floor for first 6 seconds
        if time.time() < 6:
            BASE_MOTION_NOISE = (BASE_MOTION_NOISE * 0.9) + (motion * 0.1)
            motion_threshold = BASE_MOTION_NOISE * 1.0

        prev_landmarks = lm_flat   # save for next frame

        # ignore frames during cooldown
        if cooldown_counter > 0:
            conf_history.append(0)
            cv2.putText(frame, "Cooldown...", (20, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("Realtime Gesture (ONNX)", frame)
            if cv2.waitKey(1) & 0xFF == 27:   # ESC quits
                break
            continue

        # decide: moving or still?
        if motion > motion_threshold:
            sequence.append(lm_flat)   # keep building sequence
            no_motion_count = 0
        else:
            no_motion_count += 1       # counting still frames

        # ===== GESTURE ENDED → RUN NEURAL NET =====
        if no_motion_count >= NO_MOTION_REQUIRED and len(sequence) > 20:

            seq = np.array(sequence)   # shape (N, 42)

            # pad too-short sequences with zeros
            if len(seq) < SEQ_LEN:
                pad = np.zeros((SEQ_LEN - len(seq), 42))
                seq = np.vstack([seq, pad])
            else:
                seq = seq[-SEQ_LEN:]   # take last 50 frames

            # normalize mean=0, std=1 (helps network)
            seq = (seq - np.mean(seq)) / (np.std(seq) + 1e-6)
            X = np.expand_dims(seq.astype(np.float32), axis=0)
            # X shape → (1, 50, 42) exactly what ONNX expects

            # run inference
            pred = session.run(None, {input_name: X})[0]  # returns list
            idx  = np.argmax(pred)                        # winning class
            conf = float(np.max(pred))                    # its confidence
            label = CLASSES[idx]                          # human name

            conf_history.append(conf)

            # only trust high-confidence predictions
            if conf >= 0.90:
                stable_prediction = f"{label} ({conf:.2f})"
            else:
                stable_prediction = "No gesture"

            # enter cooldown so we do not spam predictions
            cooldown_counter = COOLDOWN_FRAMES
            sequence = []           # start fresh
            prev_landmarks = None

        else:
            conf_history.append(0)  # still moving → 0 confidence

    # ============== NO HAND FOUND =========================
    else:
        prev_landmarks = None
        conf_history.append(0)

    # ============== DRAW CONFIDENCE GRAPH =================
    graph_y = 450
    graph_h = 120
    graph_w = 300
    x0 = 20

    # dark rectangle
    cv2.rectangle(frame, (x0, graph_y), (x0 + graph_w, graph_y - graph_h),
                  (50, 50, 50), 2)

    # build poly-line points
    pts = []
    for i, c in enumerate(conf_history):
        x = x0 + int((i / len(conf_history)) * graph_w)
        y = graph_y - int(c * graph_h)
        pts.append((x, y))

    # connect the dots
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], (0, 255, 0), 2)

    # ============== SHOW PREDICTION TEXT ==================
    cv2.putText(frame, display_pred, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    # ============== SHOW IMAGE ============================
    cv2.imshow("Realtime Gesture (ONNX)", frame)

    # ESC key to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==========================================================
# 6.  CLEAN-UP
# ----------------------------------------------------------
# Release camera and close windows properly.
# ==========================================================
cap.release()
cv2.destroyAllWindows()