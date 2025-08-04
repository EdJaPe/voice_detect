import os
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

# Load all .npy spectrograms
def load_data(base_dir="spectrograms", labels=["voice", "silence", "background"]):
    X = []
    y = []

    for label in labels:
        files = glob(os.path.join(base_dir, label, "*.npy"))
        for f in files:
            spec = np.load(f)
            X.append(spec)
            y.append(label)

    return np.array(X), np.array(y)

# Step 1: Load the data
X, y = load_data()

# Step 2: Encode labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 3: Reshape for CNN input (samples, height, width, channels)
X = X[..., np.newaxis]  # Add channel dimension

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Data loaded: {X.shape[0]} samples, shape: {X.shape[1:]}")
print(f"Classes: {label_encoder.classes_}")
