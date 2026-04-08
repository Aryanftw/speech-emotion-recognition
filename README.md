# 🎙️ Speech Emotion Recognition (SER)

A Machine Learning project that detects human emotions from speech/audio using extracted acoustic features.

---

## 📌 Overview

This project analyzes audio signals and predicts the **emotional state** of a speaker (e.g., happy, sad, angry, neutral) using feature extraction techniques and ML models.

---

## 🚀 Features

* 🎧 Audio feature extraction using `librosa` & `parselmouth`
* 🧠 Model training for emotion classification
* 📊 Evaluation of model performance
* 🔮 Emotion prediction from new audio input

---

## 📂 Project Structure

```
speech-emotion-recognition/
│
├── 01_extract_features.py   # Extract features from audio
├── 02_train.py              # Train ML model
├── 03_evaluate.py           # Evaluate model performance
├── 04_predict.py            # Predict emotion from audio
├── test.py                  # Testing script
├── features/                # Extracted features (ignored ideally)
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Aryanftw/speech-emotion-recognition.git
cd speech-emotion-recognition
```

---

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

---

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

⚠️ Dataset is NOT included in this repository due to size limitations.

### 🔹 You can use:

* RAVDESS dataset
* TESS dataset
* Any custom audio dataset

👉 Place your dataset inside a folder like:

```
data/
```

---

## 🔧 Usage

### Step 1: Extract Features

```bash
python 01_extract_features.py
```

---

### Step 2: Train Model

```bash
python 02_train.py
```

---

### Step 3: Evaluate Model

```bash
python 03_evaluate.py
```

---

### Step 4: Predict Emotion

```bash
python 04_predict.py
```

---

## 🧠 Technologies Used

* Python
* NumPy
* Librosa
* Parselmouth
* Scikit-learn
* PyTorch (if used)

---

## 🤝 Contributing

Feel free to fork this repo and improve it!

## 📬 Contact

If you have any questions or suggestions, feel free to connect!
