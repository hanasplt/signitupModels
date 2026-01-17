# =========================================================
# dynamic_trainer_torch_fixed_epochs_full_metrics_with_val_viz.py
# =========================================================
import os, pickle, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ---------- config ----------
EPOCHS      = 50
BATCH_SIZE  = 10
LR          = 1e-3
DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR    = "./web_demo"
ONNX_NAME   = "gesture_lstm.onnx"
ENCODER_NAME= "label_encoder.pickle"

FRAME_COUNT, VECTOR_LEN = 50, 42
# ----------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

# 1. load pickle
with open('./processed_data/dynamic_gestures_data_VISUAL.p','rb') as f:
    data_dict = pickle.load(f)

X_raw, y_raw = data_dict['data'], data_dict['labels']

# 2. keep only consistent shapes
X, y = [], []
for seq, label in zip(X_raw, y_raw):
    seq = np.array(seq, dtype=np.float32)
    if seq.shape == (FRAME_COUNT, VECTOR_LEN):
        X.append(seq)
        y.append(label)

X = np.array(X)
y = np.array(y)
print(f"Kept {len(X)} samples")

# 3. normalize per sequence
xmin = X.min(axis=(1,2), keepdims=True)
xmax = X.max(axis=(1,2), keepdims=True)
X = (X - xmin) / (xmax - xmin + 1e-6)

# 4. encode labels
le = LabelEncoder()
y_idx = le.fit_transform(y)
num_classes = len(le.classes_)
print("Classes:", le.classes_)

# 5. split data
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X, y_idx, test_size=0.30, stratify=y_idx, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

unique, counts = np.unique(y_train, return_counts=True)
print("Train distribution:")
for u, c in zip(unique, counts):
    print(le.classes_[u], c)

# 6. dataset
class SeqDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(SeqDataset(X_val, y_val), batch_size=BATCH_SIZE)

# 7. model
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

# ---------- LIVE PLOTS ----------
plt.ion()
fig, axs = plt.subplots(1, 3, figsize=(16,4))

train_losses, val_losses = [], []
train_accs, val_accs     = [], []

# 8. training loop
for epoch in range(1, EPOCHS + 1):

    # ---- TRAIN ----
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
        correct  += (out.argmax(1) == yb).sum().item()

    train_losses.append(loss_sum / len(train_loader.dataset))
    train_accs.append(correct / len(train_loader.dataset))

    # ---- VALIDATION ----
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

    # ---- LIVE METRICS PLOT ----
    axs[0].cla()
    axs[0].plot(train_losses, label='Train')
    axs[0].plot(val_losses, label='Val')
    axs[0].set_title('Loss')
    axs[0].legend()

    axs[1].cla()
    axs[1].plot(train_accs, label='Train')
    axs[1].plot(val_accs, label='Val')
    axs[1].set_ylim(0,1)
    axs[1].set_title('Accuracy')
    axs[1].legend()

    # ---- VISUALIZE VALIDATION PREDICTION ----
    idx = np.random.randint(len(X_val))
    xb = torch.tensor(X_val[idx:idx+1], dtype=torch.float32).to(DEVICE)
    yb = torch.tensor([y_val[idx]])
    xb = xb.to(DEVICE)

    logits = net(xb)
    probs = torch.softmax(logits, dim=1)
    pred = probs.argmax(1)[0].item()
    conf = probs[0, pred].item()
    true = yb[0].item()

    axs[2].cla()
    axs[2].imshow(xb[0].cpu().numpy().T, aspect='auto', cmap='viridis')
    axs[2].set_title(
        f"True: {le.classes_[true]} | "
        f"Pred: {le.classes_[pred]} ({conf:.2f})"
    )
    axs[2].set_xlabel("Frame")
    axs[2].set_ylabel("Feature")

    plt.pause(0.01)

    print(
        f"Epoch {epoch:03d} | "
        f"loss {train_losses[-1]:.4f} acc {train_accs[-1]:.3f} | "
        f"val-loss {val_losses[-1]:.4f} val-acc {val_accs[-1]:.3f}"
    )

plt.ioff()
plt.show()

# 9. save model
torch.save(net.state_dict(), os.path.join(SAVE_DIR, "final_lstm.pth"))

# 10. test evaluation
net.eval()
with torch.no_grad():
    y_pred = net(torch.tensor(X_test, dtype=torch.float32).to(DEVICE)).argmax(1).cpu()

print("\n" + classification_report(y_test, y_pred, target_names=le.classes_))

# 11. confusion matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
plt.imshow(cm, cmap='Blues')
plt.colorbar()
plt.xticks(range(num_classes), le.classes_, rotation=45)
plt.yticks(range(num_classes), le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# 12. ONNX export
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

# 13. save label encoder
with open(os.path.join(SAVE_DIR, ENCODER_NAME), 'wb') as f:
    pickle.dump(le, f)

print("✅ Training complete with live validation visualization")
