import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
import json
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "./static_models/model_static.onnx"
LABELS_PATH = "./static_models/labels.json"
CONFIDENCE_THRESHOLD = 0.75  # Only show character if > 75% confident

# Check if files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
    print(f"❌ Error: Model ({MODEL_PATH}) or Labels ({LABELS_PATH}) missing!")
    exit()

# 1. Load Labels from JSON
with open(LABELS_PATH, 'r') as f:
    labels_dict = json.load(f)

# 2. Initialize ONNX Runtime Session
# This is the engine that runs your .onnx "forest"
ort_session = ort.InferenceSession(MODEL_PATH)
input_name = ort_session.get_inputs()[0].name

# 3. Initialize MediaPipe Hand Detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# static_image_mode=False is optimized for video tracking
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.8
)

# 4. Initialize Webcam
cap = cv2.VideoCapture(0)

print("🚀 Inference Started. Press 'ESC' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame for a more natural user experience
    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape
    
    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Process only the primary hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw the hand skeleton on the frame
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Prepare landmark lists for normalization
        data_aux = []
        x_coords = []
        y_coords = []

        for lm in hand_landmarks.landmark:
            x_coords.append(lm.x)
            y_coords.append(lm.y)

        # Normalize coordinates (Subtract min to make it location-independent)
        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min(x_coords))
            data_aux.append(lm.y - min(y_coords))

        # Calculate bounding box for UI display
        x1, y1 = int(min(x_coords) * W) - 20, int(min(y_coords) * H) - 20
        x2, y2 = int(max(x_coords) * W) + 20, int(max(y_coords) * H) + 20

        # --- ONNX PREDICTION ---
        try:
            # Prepare the input tensor [1, 42]
            input_data = np.array([data_aux], dtype=np.float32)
            
            # Run the model
            # outputs[0] = class index, outputs[1] = probabilities
            outputs = ort_session.run(None, {input_name: input_data})
            
            # Extract confidence (Random Forest outputs a list of dicts for probs)
            probabilities = outputs[1][0] 
            max_conf = max(probabilities.values())
            prediction_index = str(outputs[0][0])

            # Determine Display Logic based on Threshold
            if max_conf >= CONFIDENCE_THRESHOLD:
                predicted_char = labels_dict.get(prediction_index, "?")
                display_text = f"{predicted_char} ({max_conf*100:.0f}%)"
                color = (0, 255, 0)  # Green for High Confidence
            else:
                display_text = "Analyzing..."
                color = (0, 165, 255) # Orange for Low Confidence

            # Draw the UI components
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, display_text, (x1, y1 - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Prediction error: {e}")
    else:
        # No hand detected
        cv2.putText(frame, "Waiting for hand...", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Show the result
    cv2.imshow('SignItUp - Static Inference (ONNX)', frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()