# =========================================================
# dynamic_trainer_torch.py
# =========================================================
import os, pickle, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ---------- config ----------
EPOCHS      = 100
BATCH_SIZE  = 10
LR          = 1e-3
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR    = "./trained_models"
ONNX_NAME   = "gesture_lstm.onnx"
ENCODER_NAME= "label_encoder.pickle"
# expected geometry
FRAME_COUNT, VECTOR_LEN = 50, 42   # 21 hand-points × 2 (x,y)
# ----------------------------

# 1. load pickle
with open('./processed_data/dynamic_gestures_data.p','rb') as f:
    data_dict = pickle.load(f)
X_raw, y_raw = data_dict['data'], data_dict['labels']

# 2. keep only consistent shapes
X, y = [], []
for seq, label in zip(X_raw, y_raw):
    seq = np.array(seq, dtype=np.float32)
    if seq.shape == (FRAME_COUNT, VECTOR_LEN):
        X.append(seq)
        y.append(label)
X, y = np.array(X), np.array(y)
print(f"Kept {len(X)} samples")

# 3. normalise 0-1 per sequence
xmin = X.min(axis=(1,2), keepdims=True)
xmax = X.max(axis=(1,2), keepdims=True)
X = (X - xmin)/(xmax - xmin + 1e-6)

# 4. encode labels
le = LabelEncoder()
y_idx = le.fit_transform(y)
num_classes = len(le.classes_)
print("Classes:", le.classes_)

# 5. split
X_train, X_test, y_train, y_test = train_test_split(
        X, y_idx, test_size=0.2, random_state=42, stratify=y_idx)

# 6. torch datasets
class SeqDataset(Dataset):
    def __init__(self, xx, yy):
        self.xx = torch.tensor(xx, dtype=torch.float32)
        self.yy = torch.tensor(yy, dtype=torch.long)
    def __len__(self): return len(self.xx)
    def __getitem__(self, idx): return self.xx[idx], self.yy[idx]

train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(SeqDataset(X_test,  y_test),  batch_size=BATCH_SIZE)

# 7. model
class GestureLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(VECTOR_LEN, 128, 2, batch_first=True, dropout=0.4)
        self.fc   = nn.Linear(128, num_classes)
    def forward(self, x):
        x, _ = self.lstm(x)          # [B, T, 128]
        x = x[:, -1, :]              # last timestep
        return self.fc(x)            # [B, classes]

net = GestureLSTM().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)

# 8. train loop
for epoch in range(1, EPOCHS+1):
    net.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = net(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
    # quick val acc
    net.eval()
    with torch.no_grad():
        xt, yt = torch.tensor(X_test, dtype=torch.float32).to(DEVICE), \
                 torch.tensor(y_test, dtype=torch.long).to(DEVICE)
        preds = net(xt).argmax(1).cpu()
        acc = (preds == y_test).float().mean().item()
    print(f"Epoch {epoch:03d}  val-acc {acc:.3f}")

# 9. evaluate
net.eval()
with torch.no_grad():
    y_pred = net(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).argmax(1).cpu()
print("\n"+classification_report(y_test, y_pred, target_names=le.classes_))

# 10. export to ONNX  (opset 18, no dynamo optimiser)
dummy = torch.randn(1, 50, 42).to(DEVICE)
torch.onnx.export(
        net,
        dummy,
        os.path.join(SAVE_DIR, ONNX_NAME),
        input_names=['seq'],
        output_names=['logits'],
        dynamic_axes={'seq': {0: 'batch'}},
        opset_version=18,          # ← modern, no down-grade needed
        do_constant_folding=False, # ← skip the buggy folder
        dynamo=False               # ← use stable legacy exporter
)
print(f"✅ ONNX model written to {SAVE_DIR}/{ONNX_NAME}")

# 11. save label encoder
os.makedirs(SAVE_DIR, exist_ok=True)
with open(os.path.join(SAVE_DIR, ENCODER_NAME), 'wb') as f:
    pickle.dump(le, f)