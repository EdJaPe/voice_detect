import librosa
import numpy as np
import pandas as pd
import os

# Constants
SAMPLE_RATE = 16000
DURATION = 2  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MELS = 128  # Number of Mel bands
HOP_LENGTH = 512  # Step size for FFT
FIXED_SIZE = (N_MELS, 87)  # Height x Width of spectrogram

# Load the CSV
df = pd.read_csv('dataset.csv')
X = []
y = []

def extract_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    
    # Pad or trim to exact length
    if len(y) < SAMPLES_PER_TRACK:
        y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)))
    else:
        y = y[:SAMPLES_PER_TRACK]

    # Generate Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    
    # Convert to dB scale
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Resize to fixed shape
    if mel_db.shape[1] < FIXED_SIZE[1]:
        pad_width = FIXED_SIZE[1] - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :FIXED_SIZE[1]]

    return mel_db

print("Processing audio files into spectrograms...")
for idx, row in df.iterrows():
    file_path = row['filepath']
    label = row['label']
    
    if os.path.exists(file_path):
        try:
            spectrogram = extract_mel_spectrogram(file_path)
            X.append(spectrogram)
            y.append(label)
        except Exception as e:
            print(f"Error with {file_path}: {e}")

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

print("Saving processed dataset...")
np.save('X.npy', X)
np.save('y.npy', y)

print(f"Saved: {len(X)} spectrograms with shape {X[0].shape}")
