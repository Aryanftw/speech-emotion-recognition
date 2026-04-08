import os
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call

# ───────── EMOTION MAPS ─────────
RAVDESS_EMOTIONS = {
    '01':'neutral','02':'calm','03':'happy','04':'sad',
    '05':'angry','06':'fearful','07':'disgust','08':'surprised'
}
TESS_EMOTIONS = {
    'angry':'angry','disgust':'disgust','fear':'fear',
    'happy':'happy','neutral':'neutral','ps':'surprise','sad':'sad'
}
EMODB_EMOTIONS = {
    'W':'angry','L':'boredom','E':'disgust','A':'fear',
    'F':'happy','T':'sad','N':'neutral'
}

N_FRAMES = 50


# ───────── FEATURE FUNCTION ─────────
def extract_from_signal(y, sr):
    try:
        snd = parselmouth.Sound(y, sampling_frequency=sr)
        dur = len(y) / sr

        frame_times = np.linspace(0, dur, N_FRAMES)
        hop = max(1, int(len(y) / N_FRAMES))

        pitch_obj = snd.to_pitch()
        formant_obj = call(snd, "To Formant (burg)", 0, 5, 5500, 0.025, 50)

        # Prosody-related features
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
            # pitch
            p = pitch_obj.get_value_at_time(t)
            seq[i, 0] = p if (p and not np.isnan(p)) else 0.0

            # energy
            start = i * hop
            end = min(start + hop, len(y))
            frame = y[start:end]
            seq[i, 1] = np.sqrt(np.mean(frame**2)) if len(frame) else 0.0

            # speaking rate proxy
            seq[i, 2] = onset_resampled[i]

            # formant
            f1 = call(formant_obj, "Get value at time", 1, t, 'Hertz', 'Linear')
            seq[i, 3] = f1 if (f1 and not np.isnan(f1)) else 0.0

            # extra features
            seq[i, 4] = zcr_resampled[i]
            seq[i, 5] = centroid_resampled[i]

        return seq

    except Exception as e:
        print(f"Error: {e}")
        return None


# ───────── PROCESS ONE FILE (WITH AUGMENTATION) ─────────
def process_file(path, emotion, records):
    y, sr = librosa.load(path, sr=22050)

    # ORIGINAL
    seq = extract_from_signal(y, sr)
    if seq is not None:
        records.append((seq, emotion))

    # ───── AUGMENTATIONS ─────

    # 1. Noise
    y_noise = y + np.random.normal(0, 0.005, len(y))
    seq = extract_from_signal(y_noise, sr)
    if seq is not None:
        records.append((seq, emotion))

    # 2. Time stretch
    try:
        y_stretch = librosa.effects.time_stretch(y, rate=0.9)
        seq = extract_from_signal(y_stretch, sr)
        if seq is not None:
            records.append((seq, emotion))
    except:
        pass

    # 3. Volume
    y_amp = y * 1.1
    seq = extract_from_signal(y_amp, sr)
    if seq is not None:
        records.append((seq, emotion))


# ───────── DATASET PROCESSING ─────────
def process_dataset(data_dir, emotion_map, dataset_name, structure):
    records = []

    if structure == "ravdess":
        for folder in os.listdir(data_dir):
            fp = os.path.join(data_dir, folder)
            if not os.path.isdir(fp):
                continue

            for fname in os.listdir(fp):
                if fname.endswith(".wav"):
                    emotion = emotion_map.get(fname.split("-")[2])
                    process_file(os.path.join(fp, fname), emotion, records)

    elif structure == "tess":
        for folder in os.listdir(data_dir):
            fp = os.path.join(data_dir, folder)
            if not os.path.isdir(fp):
                continue

            emotion = emotion_map.get(folder.split("_")[-1].lower())

            for fname in os.listdir(fp):
                if fname.endswith(".wav"):
                    process_file(os.path.join(fp, fname), emotion, records)

    elif structure == "emodb":
        for fname in os.listdir(data_dir):
            if fname.endswith(".wav"):
                emotion = emotion_map.get(fname[5].upper())
                process_file(os.path.join(data_dir, fname), emotion, records)

    # Save
    X = np.array([r[0] for r in records])
    y = np.array([r[1] for r in records])

    os.makedirs("features", exist_ok=True)

    np.save(f"features/{dataset_name}_sequences.npy", X)
    np.save(f"features/{dataset_name}_labels.npy", y)

    print(f"✅ {dataset_name} saved → {X.shape}")


# ───────── MAIN ─────────
if __name__ == "__main__":
    print("Processing RAVDESS...")
    process_dataset("data/RAVDESS", RAVDESS_EMOTIONS, "ravdess", "ravdess")

    print("Processing TESS...")
    process_dataset("data/TESS", TESS_EMOTIONS, "tess", "tess")

    print("Processing EMO-DB...")
    process_dataset("data/EMODB/wav", EMODB_EMOTIONS, "emodb", "emodb")