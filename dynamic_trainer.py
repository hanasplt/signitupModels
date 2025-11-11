# =========================================================
# dynamic_trainer.py — Train an LSTM model for dynamic gesture vs no-gesture
# =========================================================
# This script loads a pickled dataset of a specific gesture and a 'no gesture' class,
# normalizes the data, visualizes samples, trains an LSTM model, and saves it.
# =========================================================

# ---------------------------------------------------------
# 1. Import every library we will need later
# ---------------------------------------------------------
import os                           # Lets us talk to the operating-system (paths, folders, etc.)
import pickle                       # Lets us load / save Python objects to disk very quickly
import numpy as np                  # Fast math on arrays (the heart of ML data)
import tensorflow as tf             # Google’s deep-learning framework (Keras sits on top)
from sklearn.preprocessing import LabelEncoder        # Turns string labels → integers 0,1,2…
from sklearn.model_selection import train_test_split  # Keeps training / testing sets similar
from sklearn.utils.class_weight import compute_class_weight  # Fixes class-imbalance (more ‘no-gesture’ than ‘j’)
from sklearn.metrics import confusion_matrix, classification_report  # Pretty print of results
from tensorflow.keras.models import Sequential        # Keras “stack-of-layers” API
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization  # The actual layers
from tensorflow.keras.utils import to_categorical     # Turns integers 0,1 → one-hot vectors [1,0],[0,1]
import matplotlib.pyplot as plt     # Draw graphs, confusion matrix, live training curves, etc.

# ---------------------------------------------------------
# 2. Tunable knobs you might want to change quickly
# ---------------------------------------------------------
LIVE_PLOT = True                    # If True → pop-up window updates while training
SHOW_LANDMARKS = True               # If True → pop-up window shows skeleton animation before training
SHOW_ALL_LANDMARKS = True           # (unused flag – kept for compatibility)
SHOW_DURING_TRAIN = False           # (unused flag – kept for compatibility)
EPOCHS = 50                         # How many times we look at the entire training set
BATCH_SIZE = 8                      # How many videos we feed GPU at once

# ---------------------------------------------------------
# 3. Where are the data files and where will we save results?
# ---------------------------------------------------------
GESTURE_FILE = './processed_data/j_data.p'              # 👈 your positive-class pickle
NO_GESTURE_FILE = './processed_data/nogesture_data.p'   # 👈 your negative-class pickle

BASE_SAVE_DIR = './trained_models'                      # Top folder for all experiments
gesture_name = os.path.splitext(os.path.basename(GESTURE_FILE))[0].replace('_data', '').upper()  # Extract 'J' from j_data.p
SAVE_DIR = os.path.join(BASE_SAVE_DIR, f"{gesture_name}_vs_NOGESTURE_model")  # e.g. ./trained_models/J_vs_NOGESTURE_model
os.makedirs(SAVE_DIR, exist_ok=True)                    # Create that folder if it does not exist

print(f"📁 Models and encoder will be saved to: {SAVE_DIR}")  # Human feedback

# ---------------------------------------------------------
# 4. Helper: load a pickle that has {'data':[…], 'labels':[…]}
# ---------------------------------------------------------
def load_pickle(file_path):
    with open(file_path, 'rb') as f:        # Open file in binary-read mode
        data_dict = pickle.load(f)          # Deserialize the whole dict
    return np.array(data_dict['data'], dtype=object), np.array(data_dict['labels'])  # Return two numpy arrays

# ---------------------------------------------------------
# 5. Actually load the two pickles
# ---------------------------------------------------------
X_gesture, y_gesture = load_pickle(GESTURE_FILE)  # Videos of letter ‘J’
X_no, y_no = load_pickle(NO_GESTURE_FILE)         # Videos of “nothing”

print(f"✅ Loaded gesture samples: {len(X_gesture)}")  # Quick sanity check
print(f"✅ Loaded no-gesture samples: {len(X_no)}")

# ---------------------------------------------------------
# 6. Stick the two lists together into one big list
# ---------------------------------------------------------
X_raw = np.concatenate([X_gesture, X_no], axis=0)  # Stack video lists vertically
y_raw = np.concatenate([y_gesture, y_no], axis=0)  # Stack label lists vertically

print(f"📦 Total combined samples: {len(X_raw)}")

# ---------------------------------------------------------
# 7. Throw away videos that do not have exactly 50 frames × 42 numbers
# ---------------------------------------------------------
unique_shapes = set([np.array(x).shape for x in X_raw])  # Collect every shape we see
print(f"🔍 Found shapes in dataset: {unique_shapes}")

EXPECTED_SHAPE = (50, 42)          # (frames, features) – adjust if you changed Mediapipe settings

X, y = [], []                      # Clean lists we will actually use
for seq, label in zip(X_raw, y_raw):  # Walk through every video+label
    seq = np.array(seq)               # Ensure it is numpy array
    if seq.shape == EXPECTED_SHAPE:   # Keep only perfect rectangles
        X.append(seq)
        y.append(label)
    else:                             # Otherwise complain and skip
        print(f"⚠️ Skipping inconsistent sample with shape {seq.shape}")

X = np.array(X, dtype=np.float32)  # Convert list → 3-D float32 array
y = np.array(y)                    # Convert list → 1-D string array
print(f"✅ Kept {len(X)} valid samples")

if len(X) == 0:                    # Crash early if nothing survived
    raise ValueError("❌ No valid samples found. Adjust EXPECTED_SHAPE if needed.")

# ---------------------------------------------------------
# 8. Min-max normalise every video individually (0-1 range)
# ---------------------------------------------------------
X_min = X.min(axis=(1, 2), keepdims=True)  # Smallest number inside that video
X_max = X.max(axis=(1, 2), keepdims=True)  # Largest number inside that video
X = (X - X_min) / (X_max - X_min + 1e-6)   # Scale 0-1, avoid divide-by-zero

# ---------------------------------------------------------
# 9. Convert string labels → integers → one-hot vectors
# ---------------------------------------------------------
label_encoder = LabelEncoder()              # Instantiate scikit-learn helper
y_encoded = label_encoder.fit_transform(y)  # ['J','J','NoGesture'] → [0,0,1]
y_categorical = to_categorical(y_encoded)   # [0,0,1] → [[1,0],[1,0],[0,1]]

print(f"🎯 Classes: {list(label_encoder.classes_)}")  # Show mapping

# ---------------------------------------------------------
# 10. Split into training (80 %) and testing (20 %)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)
print(f"📊 Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

# ---------------------------------------------------------
# 11. Compute class weights so the network cares equally about ‘J’ and ‘no-gesture’
# ---------------------------------------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weights = dict(enumerate(class_weights))  # Keras wants a dict {class_id : weight}
print("⚖️ Class weights:", class_weights)

# ---------------------------------------------------------
# 12. Definition of 21 hand-bone edges (Mediapipe topology)
# ---------------------------------------------------------
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

# ---------------------------------------------------------
# 13. Function: animate one video so humans see what the network sees
# ---------------------------------------------------------
def show_landmark_sequence(sequence, label, delay=100):
    plt.figure()                                    # Create new window
    for frame in sequence:                          # Walk through 50 frames
        plt.clf()                                   # Clear previous drawing
        xs = frame[::2]                             # Even indices → x coordinates
        ys = 1 - np.array(frame[1::2])              # Odd indices → y coordinates (flip y so hand is upright)
        for a, b in HAND_CONNECTIONS:               # Draw bone lines
            plt.plot([xs[a], xs[b]], [ys[a], ys[b]], 'gray', linewidth=1)
        plt.scatter(xs, ys, c='blue', s=30)         # Draw joints
        plt.title(f"Gesture: {label}")              # Show label on top
        plt.xlim(0, 1)                              # Keep same scale
        plt.ylim(0, 1)
        plt.axis('off')                             # Remove axes
        plt.pause(0.05)                             # Short pause → animation effect
    plt.show(block=False)                           # Leave window open
    plt.pause(delay / 1000)                         # Wait a bit before closing
    plt.close()

# ---------------------------------------------------------
# 14. If user wants, quickly visualise one example per class
# ---------------------------------------------------------
if SHOW_LANDMARKS:
    print("\n🎥 Displaying gesture samples...")
    labels_to_show = np.unique(y)                   # Usually ['J', 'NoGesture']
    for lbl in labels_to_show:                      # One movie per class
        idx = np.where(y == lbl)[0][0]              # Grab first sample of that class
        show_landmark_sequence(X[idx], lbl)         # Play the mini-movie

# ---------------------------------------------------------
# 15. Build the neural network (two LSTM layers + dense head)
# ---------------------------------------------------------
model = Sequential([                                # Empty stack-of-layers
    LSTM(128, return_sequences=True, input_shape=EXPECTED_SHAPE),  # First LSTM spits whole sequence
    BatchNormalization(),                           # Normalise activations → faster training
    Dropout(0.4),                                   # Randomly ignore 40 % neurons → reduce over-fit
    LSTM(64, return_sequences=False),               # Second LSTM spits only last hidden state
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),                   # Normal fully-connected layer
    Dropout(0.3),
    Dense(len(np.unique(y)), activation='softmax')  # Output layer: probabilities that sum to 1
])

model.compile(optimizer='adam',                     # Kingma & Ba optimiser
              loss='categorical_crossentropy',      # Suitable for one-hot labels
              metrics=['accuracy'])                 # We want to see accuracy in console
model.summary()                                     # Print parameter count

# ---------------------------------------------------------
# 16. Custom callback: update a live matplotlib window while training
# ---------------------------------------------------------
class LivePlotCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        if not LIVE_PLOT:                           # Skip if user turned flag off
            return
        self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 4))  # Two sub-plots: acc & loss
        plt.ion()                                   # Interactive mode on
        self.train_acc, self.val_acc, self.train_loss, self.val_loss = [], [], [], []

    def on_epoch_end(self, epoch, logs=None):
        if LIVE_PLOT:
            self.train_acc.append(logs['accuracy'])         # Record this epoch
            self.val_acc.append(logs['val_accuracy'])
            self.train_loss.append(logs['loss'])
            self.val_loss.append(logs['val_loss'])

            self.axs[0].cla()                               # Clear left plot
            self.axs[1].cla()                               # Clear right plot
            self.axs[0].plot(self.train_acc, label='Train Acc', color='blue')
            self.axs[0].plot(self.val_acc, label='Val Acc', color='orange')
            self.axs[0].set_title('Accuracy')
            self.axs[0].legend()

            self.axs[1].plot(self.train_loss, label='Train Loss', color='green')
            self.axs[1].plot(self.val_loss, label='Val Loss', color='red')
            self.axs[1].set_title('Loss')
            self.axs[1].legend()

            plt.suptitle(f"Epoch {epoch+1}/{EPOCHS}")       # Update title
            plt.pause(0.1)                                  # Refresh window

    def on_train_end(self, logs=None):
        if LIVE_PLOT:
            plt.ioff()                                      # Leave window open when finished
            plt.show()

# ---------------------------------------------------------
# 17. Finally press the big green button: TRAIN
# ---------------------------------------------------------
history = model.fit(
    X_train, y_train,                           # Feed training data
    epochs=EPOCHS,                              # Repeat N times
    batch_size=BATCH_SIZE,                      # Mini-batch size
    validation_data=(X_test, y_test),           # Evaluate on held-out set every epoch
    class_weight=class_weights,                 # Give extra love to minority class
    callbacks=[LivePlotCallback()],             # Our live plotting
    verbose=1                                   # Print one line per epoch
)

# ---------------------------------------------------------
# 18. Save the trained brain to disk so we can load it later inside the real-time demo
# ---------------------------------------------------------
MODEL_PATH = os.path.join(SAVE_DIR, f'{gesture_name.lower()}_vs_no_gesture_lstm_model.h5')
ENCODER_PATH = os.path.join(SAVE_DIR, f'{gesture_name.lower()}_vs_no_gesture_label_encoder.pickle')

model.save(MODEL_PATH)                              # Keras HDF5 format: architecture + weights
with open(ENCODER_PATH, 'wb') as f:                 # Also save the mapping 0→'J', 1→'NoGesture'
    pickle.dump(label_encoder, f)

print(f"\n✅ Model and encoder saved to:\n  - {MODEL_PATH}\n  - {ENCODER_PATH}")

# ---------------------------------------------------------
# 19. Compute final numbers: precision, recall, F1, confusion matrix
# ---------------------------------------------------------
y_pred = model.predict(X_test)              # Soft probabilities shape (N, 2)
y_pred_labels = np.argmax(y_pred, axis=1)   # Highest probability → class index
y_true_labels = np.argmax(y_test, axis=1)   # Same for ground-truth

print("\n📊 Classification Report:")        # Nicely formatted table
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))

cm = confusion_matrix(y_true_labels, y_pred_labels)  # 2×2 matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')                # Draw heat-map
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, rotation=45)
plt.yticks(range(len(label_encoder.classes_)), label_encoder.classes_)
plt.colorbar()
plt.show()                                  # Block until user closes window