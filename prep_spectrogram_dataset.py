import os
import librosa
import numpy as np
from glob import glob

# Constants
SAMPLE_RATE = 16000
DURATION = 3  # seconds
N_MELS = 128
HOP_LENGTH = 512
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION

# Directories
input_base = "data"
output_base = "spectrograms"
labels = ["voice", "silence", "background"]

def extract_mel_spectrogram(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Pad or trim to fixed length
    if len(y) < SAMPLES_PER_FILE:
        y = np.pad(y, (0, SAMPLES_PER_FILE - len(y)))
    else:
        y = y[:SAMPLES_PER_FILE]

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    return mel_db

# Process each label directory
for label in labels:
    input_dir = os.path.join(input_base, label)
    output_dir = os.path.join(output_base, label)
    os.makedirs(output_dir, exist_ok=True)

    # Find all .wav files
    wav_files = glob(os.path.join(input_dir, "*.wav"))

    print(f"\nProcessing {label} files ({len(wav_files)} found)...")

    for file_path in wav_files:
        try:
            # Generate spectrogram
            spectrogram = extract_mel_spectrogram(file_path)

            # Build save path
            filename = os.path.splitext(os.path.basename(file_path))[0]
            save_path = os.path.join(output_dir, f"{filename}.npy")

            # Save as .npy file
            np.save(save_path, spectrogram)

            print(f"Saved: {save_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

print("\n✅ All spectrograms saved to 'spectrograms/' directory.")
