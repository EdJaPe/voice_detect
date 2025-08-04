import os

# Path to dataset
DATA_DIR="data"

# Supported classes
CLASSES = ["voice", "silence", "background"]

# Store file paths and labels
filepaths = []
labels = []


# Walk through each class folder
for label in CLASSES:
    folder_path = os.path.join(DATA_DIR, label)
    if not os.path.exists(folder_path):
    	continue
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):
            filepath = os.path.join(folder_path, filename)
            filepaths.append(filepath)
            labels.append(label)
# second check
for path, label in zip(filepaths[:5], labels[:5]):
    print(f"{path}=> {label}")

print(f"\nTotal file found: {len(filepaths)}")

