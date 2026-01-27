import pickle
import numpy as np
import os
import json
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# ONNX specific imports
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# -----------------------------
# CONFIGURATION
# -----------------------------
INPUT_FILE = './static_models/processed_static_data/static_gestures_data.p'
ONNX_OUTPUT = './static_models/model_static.onnx'
LABELS_OUTPUT = './static_models/labels.json'
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

data = np.asarray(cleaned_data).astype(np.float32) # ONNX prefers float32
raw_labels = np.asarray(cleaned_labels)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(raw_labels)
class_names = label_encoder.classes_.tolist() # Convert to list for JSON

print(f"✅ Loaded {len(data)} valid samples.")
print(f"🏷️  Classes detected: {class_names}")

# -----------------------------
# TRAIN MODEL
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    data, encoded_labels, test_size=0.2, shuffle=True, stratify=encoded_labels, random_state=42
)

print(f"🧠 Training Random Forest...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluation
y_predict = model.predict(x_test)
print(f"✨ Accuracy: {accuracy_score(y_test, y_predict) * 100:.2f}%")

# -----------------------------
# CONVERT TO ONNX
# -----------------------------
print("🔄 Converting Scikit-Learn model to ONNX...")

# Define the input shape: [BatchSize, 42 Features]
# None allows the model to process one or multiple hands at once
initial_type = [('float_input', FloatTensorType([None, EXPECTED_FEATURES]))]

onx = convert_sklearn(model, initial_types=initial_type)

with open(ONNX_OUTPUT, "wb") as f:
    f.write(onx.SerializeToString())

print(f"💾 ONNX model saved to: {ONNX_OUTPUT}")

# -----------------------------
# SAVE LABELS TO JSON
# -----------------------------
# Mapping index to label (e.g., {"0": "A", "1": "B", "2": "NG", ...})
label_map = {str(i): label for i, label in enumerate(class_names)}

with open(LABELS_OUTPUT, 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)

print(f"📄 Labels JSON saved to: {LABELS_OUTPUT}")