import pickle
import numpy as np
import os
import json
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ONNX specific imports
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# -----------------------------
# CONFIGURATION
# -----------------------------
INPUT_FILE = './static_models/processed_static_data/static_gestures_data.p'
ONNX_OUTPUT = './static_models/model_static.onnx'

# Two output paths for labels
LABELS_STATIC_JSON = './static_models/labels_static.json' # The Array format
LABELS_JSON = './static_models/labels.json'               # The Object format

EXPECTED_FEATURES = 42 

if not os.path.exists(INPUT_FILE):
    print(f"❌ Error: {INPUT_FILE} not found.")
    exit()

os.makedirs(os.path.dirname(ONNX_OUTPUT), exist_ok=True)

# -----------------------------
# LOAD & PREPARE DATA
# -----------------------------
data_dict = pickle.load(open(INPUT_FILE, 'rb'))

cleaned_data = []
cleaned_labels = []

for d, l in zip(data_dict['data'], data_dict['labels']):
    if len(d) == EXPECTED_FEATURES:
        cleaned_data.append(d)
        cleaned_labels.append(l)

data = np.asarray(cleaned_data).astype(np.float32)
raw_labels = np.asarray(cleaned_labels)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(raw_labels)
class_names = label_encoder.classes_.tolist() 

print(f"✅ Loaded {len(data)} valid samples.")
print(f"🏷️ Classes detected: {class_names}")

# -----------------------------
# TRAIN MODEL
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    data, encoded_labels, test_size=0.2, shuffle=True, stratify=encoded_labels, random_state=42
)

print(f"🧠 Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(f"✨ Accuracy: {accuracy_score(y_test, y_predict) * 100:.2f}%")

# -----------------------------
# CONVERT TO ONNX
# -----------------------------
print("🔄 Converting to ONNX...")
initial_type = [('float_input', FloatTensorType([None, EXPECTED_FEATURES]))]
onx = convert_sklearn(model, initial_types=initial_type)

with open(ONNX_OUTPUT, "wb") as f:
    f.write(onx.SerializeToString())

# -----------------------------
# SAVE LABELS (TWO VERSIONS)
# -----------------------------

# Version 1: labels_static.json -> ["1", "2V", "A", "Ñ", ...]
with open(LABELS_STATIC_JSON, 'w', encoding='utf-8') as f:
    json.dump(class_names, f, ensure_ascii=False, indent=4)
print(f"📄 Saved ARRAY format to: {LABELS_STATIC_JSON}")

# Version 2: labels.json -> {"0": "1", "1": "2V", ...}
label_map = {str(i): label for i, label in enumerate(class_names)}
with open(LABELS_JSON, 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)
print(f"📄 Saved OBJECT format to: {LABELS_JSON}")

# ----------------------------------------
# CONFUSION MATRIX
# ----------------------------------------
cm = confusion_matrix(y_test, y_predict)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

plt.figure(figsize=(12, 10))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
plt.title('Confusion Matrix', fontsize=16)
plt.ylabel('True Label', fontsize=13)
plt.xlabel('Predicted Label', fontsize=13)
plt.tight_layout()
plt.savefig('./static_models/confusion_matrix.png', dpi=150)
plt.close()
print("📊 Saved confusion matrix to: ./static_models/confusion_matrix.png")

# Also save the raw numbers as CSV (useful for LaTeX tables)
cm_df.to_csv('./static_models/confusion_matrix.csv')

# Per-class metrics (precision, recall, F1)
report = classification_report(y_test, y_predict, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('./static_models/classification_report.csv')
print("📄 Saved classification report to: ./static_models/classification_report.csv")

# ----------------------------------------
# TRAINING HISTORY (OOB Score per n_estimators)
# ----------------------------------------
print("📈 Computing training history via OOB scores...")
oob_scores = []
train_accuracies = []
estimator_range = range(1, 101, 5)  # 1, 6, 11, ... 96, 101 -> adjust as needed

for n in estimator_range:
    rf = RandomForestClassifier(n_estimators=n, oob_score=True, random_state=42)
    rf.fit(x_train, y_train)
    oob_scores.append(rf.oob_score_)
    train_acc = accuracy_score(y_train, rf.predict(x_train))
    train_accuracies.append(train_acc)

# Save history as CSV
history_df = pd.DataFrame({
    'n_estimators': list(estimator_range),
    'oob_score': oob_scores,
    'train_accuracy': train_accuracies
})
history_df.to_csv('./static_models/training_history.csv', index=False)
print("📄 Saved training history to: ./static_models/training_history.csv")

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(list(estimator_range), oob_scores, marker='o', label='OOB Score (≈ Validation)', color='steelblue')
plt.plot(list(estimator_range), train_accuracies, marker='s', label='Train Accuracy', color='coral')
plt.title('Random Forest Training History', fontsize=15)
plt.xlabel('Number of Trees (n_estimators)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1.05)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./static_models/training_history.png', dpi=150)
plt.close()
print("📊 Saved training history plot to: ./static_models/training_history.png")