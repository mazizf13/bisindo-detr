import os
from collections import Counter

def get_classes():
    return [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
        "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
        "U", "V", "W", "X", "Y", "Z"
    ]

def count_class_distribution(labels_path):
    counter = Counter()
    for label_file in os.listdir(labels_path):
        if not label_file.endswith(".txt"):
            continue
        with open(os.path.join(labels_path, label_file), "r") as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                counter[class_id] += 1
    return counter

if __name__ == "__main__":
    labels_path = "data/train/labels"
    counter = count_class_distribution(labels_path)
    classes = get_classes()
    for class_id, count in sorted(counter.items()):
        print(f"{classes[class_id]} ({class_id}): {count} samples")
