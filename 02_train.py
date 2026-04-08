import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATASET = "tess"  # change here

EPOCHS = 200
BATCH_SIZE = 64
LR = 3e-4
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.3
N_FRAMES = 50
N_FEATURES = 6


class ProsodyTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.input_proj = nn.Linear(N_FEATURES, D_MODEL)

        # safe embedding
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


# LOAD DATA
X = np.load(f"features/{DATASET}_sequences.npy")
y_str = np.load(f"features/{DATASET}_labels.npy", allow_pickle=True)

print("Shape:", X.shape)

# Normalize per sample
for i in range(len(X)):
    X[i] = (X[i] - np.mean(X[i], axis=0)) / (np.std(X[i], axis=0) + 1e-6)

le = LabelEncoder()
y = le.fit_transform(y_str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

weights = 1.0 / np.bincount(y_train)[y_train]
sampler = WeightedRandomSampler(torch.FloatTensor(weights), len(weights))

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
    batch_size=BATCH_SIZE, sampler=sampler
)

test_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
    batch_size=BATCH_SIZE
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ProsodyTransformer(len(le.classes_)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_acc = 0

for epoch in range(EPOCHS):

    model.train()
    correct = 0

    for Xb, yb in train_loader:
        Xb, yb = Xb.to(device), yb.to(device)

        # small noise augmentation
        Xb = Xb + torch.randn_like(Xb) * 0.01

        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        correct += (out.argmax(1) == yb).sum().item()

    train_acc = correct / len(train_loader.dataset)

    model.eval()
    correct = 0

    with torch.no_grad():
        for Xb, yb in test_loader:
            out = model(Xb.to(device))
            correct += (out.argmax(1) == yb.to(device)).sum().item()

    test_acc = correct / len(test_loader.dataset)

    print(f"Epoch {epoch+1}: Train {train_acc:.2f}, Test {test_acc:.2f}")

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), f"models/{DATASET}.pth")

print(f"\nBest Accuracy: {best_acc*100:.2f}%")