import os

labels_dir = "data/train/labels"
images_dir = "data/train/images"

empty_labels = []

for fname in os.listdir(labels_dir):
    if not fname.endswith(".txt"):
        continue
    
    label_path = os.path.join(labels_dir, fname)
    with open(label_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    if len(lines) == 0:
        empty_labels.append(fname)

print("Total file label kosong:", len(empty_labels))
for fname in empty_labels:
    img_name = fname.replace(".txt", ".jpg")
    print(f"- {fname}  |  image: {img_name} (cek di {images_dir})")
