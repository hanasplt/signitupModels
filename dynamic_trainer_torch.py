# =========================================================
# dynamic_trainer_torch_early_stop_full_metrics.py
# =========================================================
import os, pickle, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ---------- config ----------
EPOCHS      = 1000          # upper bound – we will stop earlier
BATCH_SIZE  = 10
LR          = 1e-3
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR    = "./web_demo"
ONNX_NAME   = "gesture_lstm.onnx"
ENCODER_NAME= "label_encoder.pickle"
# expected geometry
FRAME_COUNT, VECTOR_LEN = 50, 42   # 21 hand-points × 2 (x,y)
# early-stopping
PATIENCE    = 15
# ----------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

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

# 5. split  ->  70 % train  /  15 % val  /  15 % test
X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y_idx, test_size=0.30, random_state=42, stratify=y_idx)
X_val, X_test, y_val, y_test   = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp)

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

# 6. torch datasets
class SeqDataset(Dataset):
    def __init__(self, xx, yy):
        self.xx = torch.tensor(xx, dtype=torch.float32)
        self.yy = torch.tensor(yy, dtype=torch.long)
    def __len__(self): return len(self.xx)
    def __getitem__(self, idx): return self.xx[idx], self.yy[idx]

train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(SeqDataset(X_val,  y_val),   batch_size=BATCH_SIZE)
test_loader  = DataLoader(SeqDataset(X_test, y_test),  batch_size=BATCH_SIZE)

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

# 8. train loop  –  four curves  +  early stopping
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,4))

train_losses, train_accs = [], []
val_losses,   val_accs   = [], []

best_val_acc = 0.0
epochs_no_improve = 0
best_model_path = os.path.join(SAVE_DIR, "best_lstm.pth")

for epoch in range(1, EPOCHS+1):
    # ---------- train ----------
    net.train()
    running_loss = 0.
    running_correct = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        out = net(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        running_loss    += loss.item() * xb.size(0)
        running_correct += (out.argmax(1) == yb).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc  = running_correct / len(train_loader.dataset)
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # ---------- validation ----------
    net.eval()
    val_running_loss = 0.
    val_running_correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = net(xb)
            val_running_loss    += criterion(out, yb).item() * xb.size(0)
            val_running_correct += (out.argmax(1) == yb).sum().item()

    val_loss = val_running_loss / len(val_loader.dataset)
    val_acc  = val_running_correct / len(val_loader.dataset)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    # ---------- live plot ----------
    ax1.clear(); ax2.clear()
    # loss panel
    ax1.plot(train_losses, label='train loss', color='tab:red')
    ax1.plot(val_losses,   label='val loss',   color='tab:orange')
    ax1.legend(); ax1.set_title('Loss'); ax1.set_xlabel('epoch')
    # accuracy panel
    ax2.plot(train_accs, label='train acc', color='tab:blue')
    ax2.plot(val_accs,   label='val acc',   color='tab:cyan')
    ax2.legend(); ax2.set_title('Accuracy'); ax2.set_xlabel('epoch')
    ax2.set_ylim(0, 1)

    fig.canvas.draw(); fig.canvas.flush_events()
    print(f"Epoch {epoch:03d}  "
          f"loss {train_loss:.4f}  acc {train_acc:.3f}  |  "
          f"val-loss {val_loss:.4f}  val-acc {val_acc:.3f}")

    # ---------- early stopping ----------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(net.state_dict(), best_model_path)
    else:
        epochs_no_improve += 1

    if epochs_no_improve == PATIENCE:
        print(f"Early stopping at epoch {epoch} (best val-acc {best_val_acc:.3f})")
        break

plt.ioff(); plt.close()

# 9. restore best weights
net.load_state_dict(torch.load(best_model_path))

# 10. final evaluation on TEST set
net.eval()
with torch.no_grad():
    y_pred = net(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).argmax(1).cpu()
print("\n"+classification_report(y_test, y_pred, target_names=le.classes_))

# 11. confusion matrix plot
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, y_pred)
im = plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.colorbar(im)
plt.xticks(range(num_classes), le.classes_, rotation=45)
plt.yticks(range(num_classes), le.classes_)
plt.ylabel('True'); plt.xlabel('Predicted'); plt.title('Confusion matrix')
thresh = cm.max() / 2.
for i in range(num_classes):
    for j in range(num_classes):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'))
print(f"📊 Confusion matrix saved to {SAVE_DIR}/confusion_matrix.png")
plt.show()

# 12. export to ONNX
dummy = torch.randn(1, 50, 42).to(DEVICE)
torch.onnx.export(
        net, dummy, os.path.join(SAVE_DIR, ONNX_NAME),
        input_names=['seq'], output_names=['logits'],
        dynamic_axes={'seq':{0:'batch'}}, opset_version=18,
        do_constant_folding=False, dynamo=False
)
print(f"✅ ONNX model written to {SAVE_DIR}/{ONNX_NAME}")

# 13. save label encoder
with open(os.path.join(SAVE_DIR, ENCODER_NAME), 'wb') as f:
    pickle.dump(le, f)