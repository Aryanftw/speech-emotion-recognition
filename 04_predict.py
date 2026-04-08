import numpy as np
import torch
import torch.nn as nn
import librosa
import parselmouth
from parselmouth.praat import call
from sklearn.preprocessing import LabelEncoder

# ── CONFIG ─────────────────────
DATASET = "emodb"   # change model here
N_FRAMES = 50
N_FEATURES = 6

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.3


# ═════════ MODEL ═════════
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


# ═════════ FEATURE EXTRACTION ═════════
def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)

    snd = parselmouth.Sound(y, sampling_frequency=sr)
    dur = len(y) / sr

    frame_times = np.linspace(0, dur, N_FRAMES)
    hop = max(1, int(len(y) / N_FRAMES))

    pitch_obj = snd.to_pitch()
    formant_obj = call(snd, "To Formant (burg)", 0, 5, 5500, 0.025, 50)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_resampled = np.interp(
        np.linspace(0, len(onset_env)-1, N_FRAMES),
        np.arange(len(onset_env)),
        onset_env
    )

    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_resampled = np.interp(
        np.linspace(0, len(zcr)-1, N_FRAMES),
        np.arange(len(zcr)),
        zcr
    )

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_resampled = np.interp(
        np.linspace(0, len(centroid)-1, N_FRAMES),
        np.arange(len(centroid)),
        centroid
    )

    seq = np.zeros((N_FRAMES, 6), dtype=np.float32)

    for i, t in enumerate(frame_times):
        p = pitch_obj.get_value_at_time(t)
        seq[i, 0] = p if (p and not np.isnan(p)) else 0.0

        start = i * hop
        end = min(start + hop, len(y))
        frame = y[start:end]
        seq[i, 1] = np.sqrt(np.mean(frame**2)) if len(frame) else 0.0

        seq[i, 2] = onset_resampled[i]

        f1 = call(formant_obj, "Get value at time", 1, t, 'Hertz', 'Linear')
        seq[i, 3] = f1 if (f1 and not np.isnan(f1)) else 0.0

        seq[i, 4] = zcr_resampled[i]
        seq[i, 5] = centroid_resampled[i]

    return seq


# ═════════ LOAD MODEL ═════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint safely
state_dict = torch.load(f"models/{DATASET}.pth", map_location=device)

num_classes = state_dict['head.2.weight'].shape[0]

model = ProsodyTransformer(num_classes).to(device)
model.load_state_dict(state_dict)
model.eval()


# ═════════ LABELS (IMPORTANT) ═════════
# Load labels from dataset (same as training)
labels = np.load(f"features/{DATASET}_labels.npy", allow_pickle=True)
labels = [l for l in labels if l is not None]

le = LabelEncoder()
le.fit(labels)


# ═════════ PREDICT FUNCTION ═════════
def predict(audio_path):
    seq = extract_features(audio_path)

    # normalize (same as training)
    seq = (seq - np.mean(seq, axis=0)) / (np.std(seq, axis=0) + 1e-6)

    x = torch.FloatTensor(seq).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1)

    pred = probs.argmax(1).item()
    confidence = probs.max().item()

    emotion = le.inverse_transform([pred])[0]

    print("\n🎤 Prediction Result")
    print("----------------------------")
    print(f"Emotion    : {emotion}")
    print(f"Confidence : {confidence*100:.2f}%")

    return emotion, confidence


# ═════════ RUN ═════════
if __name__ == "__main__":
    audio_file = "data/EMODB/wav/03a05Tc.wav"
    predict(audio_file)