import os
import csv
from collections import Counter

data_dir = "data"
file_labels = []
label_counts = Counter()

# Walk through files and assign labels
for root, _, files in os.walk(data_dir):
    for f in files:
        if f.endswith(".wav"):
            path = os.path.join(root, f)

            # Label based on folder name
            if "voice" in root.lower():
                label = "voice"
            elif "silence" in root.lower():
                label = "silence"
            elif "background" in root.lower():
                label = "background"
            else:
                label = "unknown"

            file_labels.append((path, label))
            label_counts[label] += 1

            print(f"{path} => {label}")

# Print summary
print("\n--- Summary ---")
print(f"Total files found: {len(file_labels)}")
print("Count by label:")
for label, count in label_counts.items():
    print(f"{label}: {count}")

# Save to CSV
csv_path = "dataset.csv"
with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filepath", "label"])
    writer.writerows(file_labels)

print(f"\nDataset saved to {csv_path}")
