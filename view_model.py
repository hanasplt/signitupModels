import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# -------------------------------
# Load your dataset
# -------------------------------
DATA_PATH = "./processed_data/z_data.p"
print(f"📂 Loading dataset from: {DATA_PATH}")

with open(DATA_PATH, "rb") as f:
    dataset_dict = pickle.load(f)

data = np.array(dataset_dict["data"])
labels = np.array(dataset_dict["labels"])

print("✅ Dataset loaded successfully!")
print(f"📊 Data shape: {data.shape}")  # Expect (num_samples, 30, 42)
print(f"🏷️ Unique labels: {np.unique(labels)}")

# -------------------------------
# Mediapipe hand connections
# -------------------------------
mp_hands = mp.solutions.hands
connections = list(mp_hands.HAND_CONNECTIONS)

# -------------------------------
# Visualize gesture sequences
# -------------------------------
print("🎥 Visualizing gesture sequences... (Press 'q' to quit)")

for seq_idx, (sequence, label) in enumerate(zip(data, labels)):
    print(f"▶️ Showing gesture: {label} ({seq_idx + 1}/{len(data)})")

    for frame_data in sequence:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Each frame has 42 values = 21 landmarks * (x, y)
        points = [(int(frame_data[i * 2] * 640), int(frame_data[i * 2 + 1] * 480)) for i in range(21)]

        # Draw connections
        for start, end in connections:
            if start < len(points) and end < len(points):
                cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

        # Draw landmarks
        for (px, py) in points:
            cv2.circle(frame, (px, py), 5, (0, 255, 255), -1)

        # Label text
        cv2.putText(frame, f"Label: {label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Gesture Dataset Viewer", frame)

        # Wait to simulate animation speed (adjust delay if needed)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("✅ Visualization complete.")
