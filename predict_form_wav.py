import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# ---- Config ----
SAMPLE_RATE = 16000
DURATION = 3  # seconds
N_MELS = 128
HOP_LENGTH = 512
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION
LABELS = ['background', 'silence', 'voice']  # Your class order during training
MODEL_PATH = "voice_detection_model.h5"
TEST_FILE = "sel.2109.ch01.230512.195336.03.other.human voice and retrieval.wav"  # Replace with your test file path

# ---- Load and preprocess test audio ----
def preprocess_wav(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Pad or trim to exact length
    if len(y) < SAMPLES_PER_FILE:
        y = np.pad(y, (0, SAMPLES_PER_FILE - len(y)))
    else:
        y = y[:SAMPLES_PER_FILE]

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Add channel dimension
    mel_db = mel_db[..., np.newaxis]
    return mel_db

# ---- Load model ----
model = tf.keras.models.load_model(MODEL_PATH)

# ---- Encode labels ----
label_encoder = LabelEncoder()
label_encoder.fit(LABELS)

# ---- Run prediction ----
spectrogram = preprocess_wav(TEST_FILE)
spectrogram = np.expand_dims(spectrogram, axis=0)  # Shape: (1, height, width, 1)

prediction = model.predict(spectrogram)
predicted_index = np.argmax(prediction)
predicted_label = label_encoder.inverse_transform([predicted_index])[0]

print(f"\n🎤 Prediction: {predicted_label} (confidence: {prediction[0][predicted_index]:.2f})")
