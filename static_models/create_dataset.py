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
DATA_DIR = './static_models/static_data_images'     # Folder where you saved your images
OUTPUT_DIR = './static_models/processed_static_data'      # Folder for the output pickle file
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'static_gestures_data.p')
DEBUG_VISUALIZE = False              # Set True to see landmarks as they are processed
PLOT_SAMPLE = True                   # Saves a scatter plot of the hand landmarks

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# INITIALIZE MEDIAPIPE
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Using static_image_mode=True for better accuracy on individual photos
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

data = []
labels = []
class_names = []

print(f"🖼️  Scanning image folders in: {DATA_DIR}\n")

# -----------------------------
# EXTRACTION LOOP
# -----------------------------
folders = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])

for gesture_folder in folders:
    gesture_path = os.path.join(DATA_DIR, gesture_folder)
    class_names.append(gesture_folder)
    
    image_files = [f for f in os.listdir(gesture_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"📁 Processing '{gesture_folder}': Found {len(image_files)} images.")

    for img_file in image_files:
        img_path = os.path.join(gesture_path, img_file)
        img = cv2.imread(img_path)
        if img is None: continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # We only take the first hand detected (max_num_hands=1)
            hand_landmarks = results.multi_hand_landmarks[0]
            
            data_aux = []
            x_coords = []
            y_coords = []

            # Step 1: Collect all coordinates
            for lm in hand_landmarks.landmark:
                x_coords.append(lm.x)
                y_coords.append(lm.y)

            # Step 2: Normalize (Center the hand at 0,0)
            # This makes the model "location-independent"
            for i in range(len(hand_landmarks.landmark)):
                lm = hand_landmarks.landmark[i]
                data_aux.append(lm.x - min(x_coords))
                data_aux.append(lm.y - min(y_coords))

            data.append(data_aux)
            labels.append(gesture_folder)

            if DEBUG_VISUALIZE:
                mp_drawing.draw_landmarks(
                    img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )
                cv2.imshow("Extraction Debug", img)
                if cv2.waitKey(1) & 0xFF == 27: break
        else:
            print(f"   ⚠️  No hand detected in: {img_file}. Skipping.")

if DEBUG_VISUALIZE:
    cv2.destroyAllWindows()

# -----------------------------
# SAVE & SUMMARY
# -----------------------------
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump({'data': data, 'labels': labels, 'classes': class_names}, f)

print(f"\n🎯 Extraction complete!")
print(f"📦 Total samples collected: {len(data)}")
print(f"💾 Saved to: {OUTPUT_FILE}")

# -----------------------------
# OPTIONAL: VISUAL CHECK
# -----------------------------
if PLOT_SAMPLE and len(data) > 0:
    # Plotting the coordinates of the last processed hand as a scatter plot
    sample_hand = np.array(data[-1]).reshape(-1, 2)
    plt.figure(figsize=(5, 5))
    plt.scatter(sample_hand[:, 0], -sample_hand[:, 1], c='blue') # -y to flip it upright
    plt.title(f"Landmark Map: {labels[-1]}")
    plt.savefig(os.path.join(OUTPUT_DIR, "landmark_check.png"))
    print("📈 Saved visual landmark check to 'landmark_check.png'")