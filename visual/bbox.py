import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from data import DETRData
from utils.setup import get_classes, get_colors


# =========================
# Load dataset
# =========================
dataset = DETRData("data/asli/train", train=False)

CLASSES = get_classes()
COLORS = get_colors()

save_dir = "dataset_bbox_visualization"
os.makedirs(save_dir, exist_ok=True)

# =========================
# ambil 1 sample per kelas
# =========================
class_samples = {}

for img, ann in dataset:

    labels = ann["labels"]

    for label in labels:
        label = int(label)

        if label not in class_samples and label != 26:
            class_samples[label] = (img, ann)

    if len(class_samples) == 26:
        break

# urutkan A-Z
class_samples = dict(sorted(class_samples.items()))

print(f"Found {len(class_samples)} classes")

# =========================
# Visualisasi per kelas
# =========================
for cls, (img, annotations) in class_samples.items():

    fig, ax = plt.subplots(figsize=(4,4))

    img_vis = img.permute(1,2,0).numpy()
    img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min())

    ax.imshow(img_vis)

    boxes = annotations["boxes"]
    labels = annotations["labels"]

    for bbox, label in zip(boxes, labels):

        x_center, y_center, w, h = bbox.tolist()

        x_center *= 224
        y_center *= 224
        w *= 224
        h *= 224

        xmin = x_center - w/2
        ymin = y_center - h/2

        class_idx = int(label)

        color = tuple(c/255 for c in COLORS[class_idx])

        rect = Rectangle(
            (xmin, ymin),
            w,
            h,
            fill=False,
            edgecolor=color,
            linewidth=3
        )

        ax.add_patch(rect)

        ax.text(
            xmin,
            max(10, ymin-10),
            CLASSES[class_idx],
            color="white",
            fontsize=12,
            weight="bold",
            bbox=dict(facecolor=color, alpha=0.8)
        )

    ax.set_title(CLASSES[cls])
    ax.axis("off")

    save_path = os.path.join(save_dir, f"{CLASSES[cls]}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

print(f"\n✅ 26 dataset bbox images saved to: {save_dir}")