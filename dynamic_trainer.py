# =========================================================
# dynamic_trainer.py — Train an LSTM model for dynamic gestures
# =========================================================
# This script loads a pickled dataset of gesture sequences,
# cleans and normalizes the data, visualizes sample hand landmarks,
# trains an LSTM model for temporal gesture recognition,
# and saves the trained model + label encoder.
# =========================================================

import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION
# -----------------------------
LIVE_PLOT = True               # Toggle real-time training plot (accuracy/loss)
SHOW_LANDMARKS = True          # Whether to show sample hand gestures visually
SHOW_ALL_LANDMARKS = True      # Show all gesture samples or just random ones
SHOW_DURING_TRAIN = False      # Whether to display sample gestures during training
EPOCHS = 60                    # Total training epochs
BATCH_SIZE = 8                 # Mini-batch size for each training iteration

# Fix random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# LOAD DATASET
# -----------------------------
# The pickle file contains a dictionary with:
#   data  -> list of gesture sequences (each = list of frames)
#   labels -> list of corresponding gesture names
with open('./processed_data/dynamic_gestures_data.p', 'rb') as f:
    data_dict = pickle.load(f)

X_raw = np.array(data_dict['data'], dtype=object)
y_raw = np.array(data_dict['labels'])
print(f"✅ Loaded dataset: {len(X_raw)} samples")

# Check what shapes the gesture sequences have
unique_shapes = set([np.array(x).shape for x in X_raw])
print(f"🔍 Found shapes in dataset: {unique_shapes}")

# Each gesture sequence is expected to have a consistent frame count and landmark count
# Example: (50 frames, 42 landmarks) = 21 hand points × 2 (x, y)
EXPECTED_SHAPE = (50, 42)  # 👈 Adjust this if your dataset differs

# Clean inconsistent samples (only keep gestures matching EXPECTED_SHAPE)
X, y = [], []
for seq, label in zip(X_raw, y_raw):
    seq = np.array(seq)
    if seq.shape == EXPECTED_SHAPE:
        X.append(seq)
        y.append(label)
    else:
        print(f"⚠️ Skipping inconsistent sample with shape {seq.shape}")

X = np.array(X, dtype=np.float32)
y = np.array(y)
print(f"✅ Kept {len(X)} valid samples")

# Safety check — stop if no valid samples found
if len(X) == 0:
    raise ValueError("❌ No valid samples found. Adjust EXPECTED_SHAPE to match dataset structure.")

# Normalize each sequence between 0–1
# This helps the model learn better since input values are standardized
X_min = X.min(axis=(1, 2), keepdims=True)
X_max = X.max(axis=(1, 2), keepdims=True)
X = (X - X_min) / (X_max - X_min + 1e-6)

# Encode text labels (e.g., "J", "Z") → numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

print(f"🎯 Classes: {list(label_encoder.classes_)}")

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)
print(f"📊 Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

# Compute class weights to handle class imbalance (if some gestures have fewer samples)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_encoded),
    y=y_encoded
)
class_weights = dict(enumerate(class_weights))
print("⚖️ Class weights:", class_weights)

# -----------------------------
# HAND LANDMARK VISUALIZATION
# -----------------------------
# Used to visualize gesture sequences by connecting hand keypoints
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def show_landmark_sequence(sequence, label, delay=100):
    """Displays a single gesture sequence frame-by-frame."""
    plt.figure()
    for frame in sequence:
        plt.clf()
        xs = frame[::2]
        ys = 1 - np.array(frame[1::2])  # Flip vertically for better view
        # Draw the hand skeleton
        for a, b in HAND_CONNECTIONS:
            plt.plot([xs[a], xs[b]], [ys[a], ys[b]], 'gray', linewidth=1)
        plt.scatter(xs, ys, c='blue', s=30)
        plt.title(f"Gesture: {label}")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.pause(0.05)
    plt.show(block=False)
    plt.pause(delay / 1000)
    plt.close()

# Show a few gesture samples before training (for confirmation)
if SHOW_LANDMARKS:
    print("\n🎥 Displaying gesture samples...")
    if SHOW_ALL_LANDMARKS:
        for lbl in np.unique(y):
            idx = np.where(y == lbl)[0][0]
            show_landmark_sequence(X[idx], lbl)
    else:
        sample_indices = np.random.choice(len(X), 3, replace=False)
        for idx in sample_indices:
            show_landmark_sequence(X[idx], y[idx])

# -----------------------------
# DEFINE MODEL (LSTM)
# -----------------------------
# LSTM (Long Short-Term Memory) is ideal for dynamic gestures
# because it learns patterns across time (i.e., movement sequences)
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=EXPECTED_SHAPE),
    BatchNormalization(),
    Dropout(0.4),
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation='softmax')  # Output layer for class probabilities
])

# Compile model with Adam optimizer and categorical cross-entropy loss
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# LIVE TRAINING VISUALIZATION
# -----------------------------
class LivePlotCallback(tf.keras.callbacks.Callback):
    """Custom callback to show real-time accuracy/loss plots during training."""
    def on_train_begin(self, logs=None):
        if not LIVE_PLOT:
            return
        self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 4))
        plt.ion()
        self.train_acc, self.val_acc, self.train_loss, self.val_loss = [], [], [], []

    def on_epoch_end(self, epoch, logs=None):
        if LIVE_PLOT:
            self.train_acc.append(logs['accuracy'])
            self.val_acc.append(logs['val_accuracy'])
            self.train_loss.append(logs['loss'])
            self.val_loss.append(logs['val_loss'])

            # Update plots each epoch
            self.axs[0].cla()
            self.axs[1].cla()
            self.axs[0].plot(self.train_acc, label='Train Acc', color='blue')
            self.axs[0].plot(self.val_acc, label='Val Acc', color='orange')
            self.axs[0].set_title('Accuracy')
            self.axs[0].legend()

            self.axs[1].plot(self.train_loss, label='Train Loss', color='green')
            self.axs[1].plot(self.val_loss, label='Val Loss', color='red')
            self.axs[1].set_title('Loss')
            self.axs[1].legend()

            plt.suptitle(f"Epoch {epoch+1}/{EPOCHS}")
            plt.pause(0.1)

        # Optionally show sample gestures every 10 epochs
        if SHOW_DURING_TRAIN and (epoch % 10 == 0):
            idx = np.random.randint(0, len(X_train))
            label_name = label_encoder.inverse_transform([np.argmax(y_train[idx])])[0]
            show_landmark_sequence(X_train[idx], label_name)

    def on_train_end(self, logs=None):
        if LIVE_PLOT:
            plt.ioff()
            plt.show()

# -----------------------------
# TRAIN MODEL
# -----------------------------
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[LivePlotCallback()],
    verbose=1
)

# -----------------------------
# SAVE MODEL & LABEL ENCODER
# -----------------------------
# Folder where model + encoder will be saved
SAVE_DIR = "./trained_models"   # 👈 Change this path to wherever you want
os.makedirs(SAVE_DIR, exist_ok=True)

# Define full file paths
MODEL_PATH = os.path.join(SAVE_DIR, "gesture_lstm_model.h5")
ENCODER_PATH = os.path.join(SAVE_DIR, "label_encoder.pickle")

# Save model and label encoder
model.save(MODEL_PATH)
with open(ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

print(f"\n✅ Model training complete and saved to:\n   🧠 Model: {MODEL_PATH}\n   🏷️ Label Encoder: {ENCODER_PATH}")


# -----------------------------
# EVALUATE PERFORMANCE
# -----------------------------
# Predict on test set and show accuracy, precision, recall, and F1-score
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))

# Plot confusion matrix to visualize which gestures are confused with others
cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_, rotation=45)
plt.yticks(range(len(label_encoder.classes_)), label_encoder.classes_)
plt.colorbar()
plt.show()