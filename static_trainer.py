# train_random_forest_v2.py
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_FILE = 'j_z.p'          # Your extracted dataset
MODEL_FILE = 'rf_model_j_z.p'
FLATTEN = True                # Flatten sequences for RandomForest (no temporal model)
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200            # Number of trees in the forest

# -----------------------------
# LOAD DATA
# -----------------------------
print(f"📦 Loading dataset from {DATA_FILE}...")
with open(DATA_FILE, 'rb') as f:
    dataset = pickle.load(f)

data = np.array(dataset['data'])
labels = np.array(dataset['labels'])

print(f"✅ Loaded {len(data)} samples | Sequence length: {data.shape[1]} | Features per frame: {data.shape[2]}")

# -----------------------------
# CLEAN INCONSISTENT SAMPLES
# -----------------------------
EXPECTED_FEATURES = data.shape[2]
clean_data, clean_labels = [], []

for d, l in zip(data, labels):
    if d.shape[1] == EXPECTED_FEATURES:
        clean_data.append(d)
        clean_labels.append(l)
    else:
        print(f"⚠️ Skipping sample with shape {d.shape}")

data = np.array(clean_data)
labels = np.array(clean_labels)

# -----------------------------
# PREPROCESS: Flatten if needed
# -----------------------------
if FLATTEN:
    # Flatten sequence into 1D vector per sample (30×42 → 1260 features)
    n_samples, seq_len, features = data.shape
    data = data.reshape(n_samples, seq_len * features)
    print(f"📉 Flattened data shape: {data.shape}")
else:
    print("⚙️ Keeping 3D sequences (for temporal models)")

# -----------------------------
# SPLIT DATA
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=TEST_SIZE, stratify=labels, random_state=RANDOM_STATE
)

print(f"🧩 Training samples: {len(x_train)}, Testing samples: {len(x_test)}")

# -----------------------------
# TRAIN MODEL
# -----------------------------
print("\n🌲 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
model.fit(x_train, y_train)

# -----------------------------
# EVALUATE MODEL
# -----------------------------
print("\n📊 Evaluating model...")
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
print("\nDetailed classification report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_test, y_pred, labels=np.unique(labels))
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(labels),
            yticklabels=np.unique(labels))
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# -----------------------------
# SAVE MODEL
# -----------------------------
with open(MODEL_FILE, 'wb') as f:
    pickle.dump({'model': model}, f)

print(f"\n💾 Model saved to {MODEL_FILE}")
