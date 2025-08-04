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

# Step 5: Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')  # 3 classes
])

# Step 6: Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Step 7: Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=8
)

# Step 8: Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc:.2f}")

# Optional: Save model
model.save("voice_detection_model.h5")
