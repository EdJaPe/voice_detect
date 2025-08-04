import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Set this to your spectrogram folder
BASE_DIR = "spectrograms"
LABELS = ["voice", "silence", "background"]

# How many to show per label
SAMPLES_PER_LABEL = 3

for label in LABELS:
    files = glob(os.path.join(BASE_DIR, label, "*.npy"))[:SAMPLES_PER_LABEL]

    for i, file_path in enumerate(files):
        spectrogram = np.load(file_path)

        plt.figure(figsize=(6, 4))
        plt.imshow(spectrogram, origin='lower', aspect='auto', cmap='magma')
        plt.title(f"{label.upper()} - {os.path.basename(file_path)}")
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        plt.show()
