import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── CONFIG ─────────────────────────
DATASET = "ravdess"   # change here

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.3
N_FEATURES = 6


# ═════════ MODEL (SAME AS TRAIN) ═════════
class ProsodyTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.input_proj = nn.Linear(N_FEATURES, D_MODEL)
        self.pos_embedding = nn.Embedding(100, D_MODEL)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=128,
            dropout=DROPOUT,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=N_LAYERS
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, D_MODEL))

        self.head = nn.Sequential(
            nn.LayerNorm(D_MODEL),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, num_classes)
        )

    def forward(self, x):
        B = x.size(0)

        x = self.input_proj(x)

        pos = torch.arange(x.size(1), device=x.device)
        pos = torch.clamp(pos, max=99)
        x = x + self.pos_embedding(pos)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        x = self.transformer(x)

        return self.head(x[:, 0])


# ═════════ LOAD DATA ═════════
X = np.load(f"features/{DATASET}_sequences.npy")
y_str = np.load(f"features/{DATASET}_labels.npy", allow_pickle=True)

print("Loaded:", X.shape)

# Normalize SAME AS TRAIN
for i in range(len(X)):
    X[i] = (X[i] - np.mean(X[i], axis=0)) / (np.std(X[i], axis=0) + 1e-6)

# REMOVE None labels
mask = np.array([e is not None for e in y_str])

X = X[mask]
y_str = y_str[mask]

le = LabelEncoder()
y = le.fit_transform(y_str)

# SAME SPLIT AS TRAIN
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ═════════ LOAD MODEL ═════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD STATE FIRST TO GET CORRECT SHAPE
state_dict = torch.load(f"models/{DATASET}.pth", map_location=device)

num_classes = state_dict['head.2.weight'].shape[0]

model = ProsodyTransformer(num_classes).to(device)
model.load_state_dict(state_dict)
model.eval()

# ═════════ PREDICT ═════════
X_tensor = torch.FloatTensor(X_test).to(device)

with torch.no_grad():
    outputs = model(X_tensor)
    preds = outputs.argmax(1).cpu().numpy()

# ═════════ RESULTS ═════════
acc = accuracy_score(y_test, preds) * 100

print("\n" + "="*40)
print(f"{DATASET.upper()} Accuracy: {acc:.2f}%")
print("="*40)

print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=le.classes_))

# ═════════ CONFUSION MATRIX ═════════
cm = confusion_matrix(y_test, preds)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap='Blues')

plt.title(f"{DATASET.upper()} Confusion Matrix\nAccuracy: {acc:.2f}%")
plt.xlabel("Predicted")
plt.ylabel("True")

plt.tight_layout()
plt.savefig(f"{DATASET}_confusion.png", dpi=150)
plt.show()