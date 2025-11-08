# =========================================================
# dynamic_trainer.py — Train an LSTM model for dynamic gesture vs no-gesture
# =========================================================
# This script loads a pickled dataset of a specific gesture and a 'no gesture' class,
# normalizes the data, visualizes samples, trains an LSTM model, and saves it.
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
LIVE_PLOT = True
SHOW_LANDMARKS = True
SHOW_ALL_LANDMARKS = True
SHOW_DURING_TRAIN = False
EPOCHS = 60
BATCH_SIZE = 8

# -----------------------------
# PATH SETTINGS
# -----------------------------
GESTURE_FILE = './processed_data/hello_data.p'       # 👈 Your main gesture
NO_GESTURE_FILE = './processed_data/nogesture_data.p'  # 👈 The “no gesture” dataset

BASE_SAVE_DIR = './trained_models'
gesture_name = os.path.splitext(os.path.basename(GESTURE_FILE))[0].replace('_data', '').upper()
SAVE_DIR = os.path.join(BASE_SAVE_DIR, f"{gesture_name}_vs_NOGESTURE_model")
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"📁 Models and encoder will be saved to: {SAVE_DIR}")

# -----------------------------
# LOAD DATASETS
# -----------------------------
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
    return np.array(data_dict['data'], dtype=object), np.array(data_dict['labels'])

X_gesture, y_gesture = load_pickle(GESTURE_FILE)
X_no, y_no = load_pickle(NO_GESTURE_FILE)

print(f"✅ Loaded gesture samples: {len(X_gesture)}")
print(f"✅ Loaded no-gesture samples: {len(X_no)}")

# Combine both
X_raw = np.concatenate([X_gesture, X_no], axis=0)
y_raw = np.concatenate([y_gesture, y_no], axis=0)

print(f"📦 Total combined samples: {len(X_raw)}")

# -----------------------------
# CLEAN & VALIDATE SHAPES
# -----------------------------
unique_shapes = set([np.array(x).shape for x in X_raw])
print(f"🔍 Found shapes in dataset: {unique_shapes}")

EXPECTED_SHAPE = (50, 42)  # (frames, features)

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

if len(X) == 0:
    raise ValueError("❌ No valid samples found. Adjust EXPECTED_SHAPE if needed.")

# Normalize per sequence
X_min = X.min(axis=(1, 2), keepdims=True)
X_max = X.max(axis=(1, 2), keepdims=True)
X = (X - X_min) / (X_max - X_min + 1e-6)

# -----------------------------
# ENCODE LABELS
# -----------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

print(f"🎯 Classes: {list(label_encoder.classes_)}")

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

print(f"📊 Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

# Compute class weights (handle imbalance)
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
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def show_landmark_sequence(sequence, label, delay=100):
    plt.figure()
    for frame in sequence:
        plt.clf()
        xs = frame[::2]
        ys = 1 - np.array(frame[1::2])
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

if SHOW_LANDMARKS:
    print("\n🎥 Displaying gesture samples...")
    labels_to_show = np.unique(y)
    for lbl in labels_to_show:
        idx = np.where(y == lbl)[0][0]
        show_landmark_sequence(X[idx], lbl)

# -----------------------------
# DEFINE MODEL (LSTM)
# -----------------------------
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=EXPECTED_SHAPE),
    BatchNormalization(),
    Dropout(0.4),
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# LIVE TRAINING VISUALIZATION
# -----------------------------
class LivePlotCallback(tf.keras.callbacks.Callback):
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
MODEL_PATH = os.path.join(SAVE_DIR, f'{gesture_name.lower()}_vs_no_gesture_lstm_model.h5')
ENCODER_PATH = os.path.join(SAVE_DIR, f'{gesture_name.lower()}_vs_no_gesture_label_encoder.pickle')

model.save(MODEL_PATH)
with open(ENCODER_PATH, 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"\n✅ Model and encoder saved to:\n  - {MODEL_PATH}\n  - {ENCODER_PATH}")

# -----------------------------
# EVALUATE PERFORMANCE
# -----------------------------
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))

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
