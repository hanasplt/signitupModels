import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# -------------------------------
# Load your dataset
# -------------------------------
DATA_PATH = "processed_video_dataset.pkl"  # CHANGE IF NEEDED
print(f"📂 Loading dataset from: {DATA_PATH}")

with open(DATA_PATH, "rb") as f:
    dataset_dict = pickle.load(f)

data = dataset_dict["X"]
labels = dataset_dict["y"]
classes = dataset_dict["classes"]

print("✅ Dataset loaded successfully!")
print(f"📊 Number of sequences: {len(data)}")
print(f"🏷️ Classes: {classes}")

# -------------------------------
# Mediapipe hand connections
# -------------------------------
mp_hands = mp.solutions.hands
connections = list(mp_hands.HAND_CONNECTIONS)

# -------------------------------
# Visualization settings
# -------------------------------
CANVAS_SIZE = 500  # square canvas
PADDING = 80       # avoid touching edges

print("🎥 Visualizing gesture sequences... (Press 'q' to quit)")

def normalize_and_center(points):
    """Normalize hand landmarks to fit centered inside a square canvas."""

    points = np.array(points, dtype=np.float32)  # shape: (21, 2)

    xs = points[:, 0]
    ys = points[:, 1]

    # Get bounding box
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)

    # Width and height of hand bbox
    w = max_x - min_x
    h = max_y - min_y

    # Avoid division by zero
    if w == 0: w = 1e-6
    if h == 0: h = 1e-6

    # Normalize relative to bounding box
    xs = (xs - min_x) / w
    ys = (ys - min_y) / h

    # Scale to canvas with padding
    xs = xs * (CANVAS_SIZE - 2 * PADDING) + PADDING
    ys = ys * (CANVAS_SIZE - 2 * PADDING) + PADDING

    return list(zip(xs.astype(int), ys.astype(int)))

for seq_idx, (sequence, label) in enumerate(zip(data, labels)):
    print(f"▶️ Showing gesture: {label} ({seq_idx + 1}/{len(data)})")

    for frame_data in sequence:

        # frame_data contains: [x1, y1, z1, x2, y2, z2 ...]
        # Extract only x, y
        xy = []
        for i in range(21):
            x = frame_data[i * 3]
            y = frame_data[i * 3 + 1]
            xy.append((x, y))

        # Normalize + center
        centered_points = normalize_and_center(xy)

        # Create black canvas
        frame = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)

        # Draw connections
        for start, end in connections:
            cv2.line(frame, centered_points[start], centered_points[end], (0, 255, 0), 2)

        # Draw landmarks
        for (px, py) in centered_points:
            cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)

        # Label text
        cv2.putText(frame, f"Label: {classes[label]}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Gesture Dataset Viewer", frame)

        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("✅ Visualization complete.")
