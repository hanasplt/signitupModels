# =========================================================
# LSTM TRAINER WITH SEMANTIC HAND FEATURES (RESEARCH-GRADE)
# =========================================================

import os, pickle, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
EPOCHS = 50
BATCH_SIZE = 10
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

SAVE_DIR = "./web_demo/PHRASE"
ONNX_NAME = "gesture_lstm_PHRASES.onnx"
ENCODER_NAME = "label_encoder_PHRASES.pickle"

FRAME_COUNT = 50

RAW_LM = 42
VEL = 42
FSTATE = 5
VECTOR_LEN = RAW_LM + VEL + FSTATE   # 89

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# 1. LOAD DATA
# =========================================================

with open('./processed_data/dynamic_gestures_data_PHRASE.p','rb') as f:
    data_dict = pickle.load(f)

X_raw, y_raw = data_dict['data'], data_dict['labels']

# =========================================================
# 2. FEATURE ENGINEERING
# =========================================================

def extract_semantic_features(seq):
    seq = np.array(seq, dtype=np.float32)

    velocities = np.diff(seq, axis=0, prepend=seq[:1])

    finger_states_seq = []
    for frame in seq:
        lm = frame.reshape(21, 2)

        states = [
            lm[4,1]  < lm[3,1],
            lm[8,1]  < lm[6,1],
            lm[12,1] < lm[10,1],
            lm[16,1] < lm[14,1],
            lm[20,1] < lm[18,1],
        ]
        finger_states_seq.append(states)

    finger_states_seq = np.array(finger_states_seq, dtype=np.float32)

    return np.concatenate([seq, velocities, finger_states_seq], axis=1)

X, y = [], []

for seq, label in zip(X_raw, y_raw):
    seq = np.array(seq, dtype=np.float32)
    if seq.shape == (FRAME_COUNT, RAW_LM):
        X.append(extract_semantic_features(seq))
        y.append(label)

X = np.array(X)
y = np.array(y)

print("Final feature shape:", X.shape)

# =========================================================
# 3. NORMALIZATION (FIXED FOR WEB APP COMPATIBILITY)
# =========================================================

# Calculate mean and std across all samples and all frames (axis 0 and 1)
# This results in one value per feature (89 total)
mean = X.mean(axis=(0, 1)) 
std = X.std(axis=(0, 1)) + 1e-6

# Apply normalization to the training data
X = (X - mean) / std

# Save as .npy for Python/Research use
np.save(os.path.join(SAVE_DIR, "norm_mean_PHRASES.npy"), mean)
np.save(os.path.join(SAVE_DIR, "norm_std_PHRASES.npy"), std)

# Save as .json for the Web App (ONNX Runtime)
# We use .tolist() to ensure it is a clean JSON array
import json

with open(os.path.join(SAVE_DIR, "norm_mean_PHRASES.json"), "w") as f:
    json.dump(mean.tolist(), f)

with open(os.path.join(SAVE_DIR, "norm_std_PHRASES.json"), "w") as f:
    json.dump(std.tolist(), f)

print(f"✅ Normalization assets saved to {SAVE_DIR}")

# =========================================================
# 4. LABEL ENCODING
# =========================================================

le = LabelEncoder()
y_idx = le.fit_transform(y)
num_classes = len(le.classes_)
print("Classes:", le.classes_)

# =========================================================
# 5. SPLIT (550 SAMPLES BALANCED)
# =========================================================

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y_idx, test_size=0.30, stratify=y_idx, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

print(f"Train {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

# =========================================================
# 6. DATASET
# =========================================================

class SeqDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(SeqDataset(X_val, y_val), batch_size=BATCH_SIZE)

# =========================================================
# 7. MODEL
# =========================================================

class GestureLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(VECTOR_LEN, 128, 2, batch_first=True, dropout=0.4)
        self.fc   = nn.Linear(128, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])

net = GestureLSTM().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

# =========================================================
# 8. LIVE TRAINING VISUALIZATION
# =========================================================

plt.ion()
fig, axs = plt.subplots(1, 3, figsize=(16,4))

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(1, EPOCHS + 1):

    net.train()
    loss_sum, correct = 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = net(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * xb.size(0)
        correct += (out.argmax(1) == yb).sum().item()

    train_losses.append(loss_sum / len(train_loader.dataset))
    train_accs.append(correct / len(train_loader.dataset))

    net.eval()
    v_loss, v_correct = 0, 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = net(xb)
            v_loss += criterion(out, yb).item() * xb.size(0)
            v_correct += (out.argmax(1) == yb).sum().item()

    val_losses.append(v_loss / len(val_loader.dataset))
    val_accs.append(v_correct / len(val_loader.dataset))

    axs[0].cla()
    axs[0].plot(train_losses, label='Train')
    axs[0].plot(val_losses, label='Val')
    axs[0].set_title("Loss")
    axs[0].legend()

    axs[1].cla()
    axs[1].plot(train_accs, label='Train')
    axs[1].plot(val_accs, label='Val')
    axs[1].set_ylim(0,1)
    axs[1].set_title("Accuracy")
    axs[1].legend()

    idx = np.random.randint(len(X_val))
    xb = torch.tensor(X_val[idx:idx+1], dtype=torch.float32).to(DEVICE)
    logits = net(xb)
    probs = torch.softmax(logits, 1)
    pred = probs.argmax(1).item()
    conf = probs[0,pred].item()
    true = y_val[idx]

    axs[2].cla()
    axs[2].imshow(xb[0].cpu().numpy().T, aspect='auto', cmap='viridis')
    axs[2].set_title(f"True: {le.classes_[true]} | Pred: {le.classes_[pred]} ({conf:.2f})")

    plt.pause(0.01)

    print(f"Epoch {epoch:03d} | "
          f"loss {train_losses[-1]:.4f} acc {train_accs[-1]:.3f} | "
          f"val-loss {val_losses[-1]:.4f} val-acc {val_accs[-1]:.3f}")

plt.ioff()
plt.savefig(os.path.join(SAVE_DIR, "training_history_PHRASES.jpg"), format='jpg', dpi=150, bbox_inches='tight')
plt.show()

# =========================================================
# 10. TEST SET EVALUATION (CRITICAL)
# =========================================================

net.eval()
with torch.no_grad():
    y_pred = net(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)) \
                .argmax(1).cpu().numpy()

print("\nTEST CLASSIFICATION REPORT:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# =========================================================
# 11. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7,6))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix (Test Set)")
plt.colorbar()

plt.xticks(range(num_classes), le.classes_, rotation=45)
plt.yticks(range(num_classes), le.classes_)

plt.xlabel("Predicted")
plt.ylabel("True")

for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, cm[i, j],
                 ha="center", va="center",
                 color="white" if cm[i,j] > cm.max()/2 else "black")

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix_PHRASES.jpg"), format='jpg', dpi=150, bbox_inches='tight')
plt.show()

# =========================================================
# 12. EXPORT ONNX
# =========================================================

dummy = torch.randn(1, FRAME_COUNT, VECTOR_LEN, device=DEVICE)
torch.onnx.export(
    net, dummy,
    os.path.join(SAVE_DIR, ONNX_NAME),
    input_names=['seq'],
    output_names=['logits'],
    dynamic_axes={'seq': {0: 'batch'}, 'logits': {0: 'batch'}},
    opset_version=17,
    do_constant_folding=False,
    dynamo=False
)

with open(os.path.join(SAVE_DIR, ENCODER_NAME), 'wb') as f:
    pickle.dump(le, f)

# Save label classes as JSON for web app compatibility
with open(os.path.join(SAVE_DIR, "labels_PHRASES.json"), "w") as f:
    json.dump(le.classes_.tolist(), f)

print("✅ Semantic-feature model trained, evaluated, and exported")
