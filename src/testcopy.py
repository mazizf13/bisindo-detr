# # ========== AZXX ==========
# import os
# import torch
# import time
# from matplotlib import pyplot as plt
# from matplotlib.patches import Rectangle

# from data import DETRData
# from model import DETR
# from utils.boxes import rescale_bboxes
# from utils.setup import get_classes, get_colors
# from utils.logger import get_logger
# from utils.rich_handlers import TestHandler, DetectionHandler


# # =========================
# # Logger
# # =========================
# logger = get_logger("test")
# test_handler = TestHandler()
# detection_handler = DetectionHandler()

# logger.print_banner()

# # =========================
# # Device
# # =========================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logger.test(f"Running on device: {device}")

# # =========================
# # Dataset
# # =========================
# test_dataset = DETRData("data/asli/test", train=False)

# # =========================
# # Model
# # =========================
# num_classes = 26
# model = DETR(num_classes=num_classes)
# model = model.to(device)
# model.eval()

# model.load_pretrained(
#     "pretrained/warnet2/noaug/300_model.pt",
#     map_location=device
# )

# logger.success("Model loaded successfully")

# # =========================
# # Classes & Colors
# # =========================
# CLASSES = get_classes()
# COLORS = get_colors()

# # =========================
# # Save directory
# # =========================
# save_dir = "test_predictions/noaug"
# os.makedirs(save_dir, exist_ok=True)

# # =========================
# # Ambil 1 sample per kelas
# # =========================
# class_samples = {}

# for img, ann in test_dataset:

#     labels = ann["labels"]

#     for label in labels:
#         label = int(label)

#         if label not in class_samples and label != 26:
#             class_samples[label] = (img, ann)

#     if len(class_samples) == 26:
#         break

# # urutkan A-Z
# class_samples = dict(sorted(class_samples.items()))

# logger.test("Running inference for 26 alphabet classes")

# # =========================
# # Inference per gambar
# # =========================
# with torch.no_grad():

#     for cls, (img, ann) in class_samples.items():

#         X = img.unsqueeze(0).to(device)

#         start_time = time.time()
#         result = model(X)
#         inference_time = (time.time() - start_time) * 1000

#         probabilities = result["pred_logits"].softmax(-1)[0, :, :-1]
#         max_probs, max_classes = probabilities.max(-1)

#         top_score, top_idx = max_probs.max(0)

#         fig, ax = plt.subplots(figsize=(4, 4))

#         ax.imshow(img.permute(1, 2, 0))

#         if top_score > 0.7:

#             pred_box = result["pred_boxes"][0, top_idx].unsqueeze(0)
#             bbox = rescale_bboxes(pred_box, (224, 224))[0]

#             pred_class = int(max_classes[top_idx])
#             score = float(top_score)

#             xmin, ymin, xmax, ymax = bbox.detach().cpu().numpy()

#             color = tuple(float(c) / 255 for c in COLORS[pred_class])

#             ax.add_patch(
#                 Rectangle(
#                     (xmin, ymin),
#                     xmax - xmin,
#                     ymax - ymin,
#                     fill=False,
#                     color=color,
#                     linewidth=3
#                 )
#             )

#             text = f"{CLASSES[pred_class]}: {score:.2f}"

#             ax.text(
#                 xmin,
#                 ymin,
#                 text,
#                 fontsize=12,
#                 color="white",
#                 bbox=dict(facecolor=color, edgecolor="none", alpha=0.6)
#             )

#         ax.set_title(f"GT: {CLASSES[cls]}")
#         ax.axis("off")

#         save_path = os.path.join(save_dir, f"{CLASSES[cls]}.png")
#         plt.savefig(save_path, bbox_inches="tight")

#         plt.close()

#         detection_handler.log_inference_time(inference_time)

# print(f"\n✅ 26 prediction images saved to: {save_dir}")



# ========== AMBIL HURUF A AJA ==========
import torch
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from data import DETRData
from model import DETR
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# =========================
# Dataset
# =========================
test_dataset = DETRData("data/asli/test", train=False)

# =========================
# Model
# =========================
model = DETR(num_classes=26).to(device)
model.eval()

model.load_pretrained(
    # "pretrained/warnet2/noaug/300_model.pt",
    "pretrained/warnet2/aug-new/300_model.pt",
    map_location=device
)

print("Model loaded successfully")

CLASSES = get_classes()
COLORS = get_colors()

# =========================
# Cari SEMUA sampel huruf A
# =========================
target_class = 0  # A biasanya index 0
best_sample = None
best_score_global = 0
best_result = None
best_img = None

# =========================
# Loop semua data
# =========================
with torch.no_grad():
    for img, ann in test_dataset:

        labels = ann["labels"]

        # cek apakah ada huruf A
        if target_class not in labels:
            continue

        X = img.unsqueeze(0).to(device)

        start_time = time.time()
        result = model(X)
        inference_time = (time.time() - start_time) * 1000

        probabilities = result["pred_logits"].softmax(-1)[0, :, :-1]
        max_probs, max_classes = probabilities.max(-1)

        top_score, top_idx = max_probs.max(0)
        pred_class = int(max_classes[top_idx])

        # hanya ambil prediksi A
        if pred_class == target_class and top_score > best_score_global:
            best_score_global = float(top_score)
            best_sample = (img, ann)
            best_result = result
            best_img = img

# =========================
# Tampilkan hasil terbaik
# =========================
if best_sample is not None:

    print("\n=== RESULT ===")
    print(f"Confidence Score: {best_score_global:.4f}")

    img = best_img
    result = best_result

    probabilities = result["pred_logits"].softmax(-1)[0, :, :-1] # type: ignore
    max_probs, max_classes = probabilities.max(-1)
    top_score, top_idx = max_probs.max(0)

    pred_box = result["pred_boxes"][0, top_idx].unsqueeze(0) # type: ignore
    bbox = rescale_bboxes(pred_box, (224, 224))[0]

    pred_class = int(max_classes[top_idx])

    xmin, ymin, xmax, ymax = bbox.detach().cpu().numpy()

    color = tuple(float(c) / 255 for c in COLORS[pred_class])

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img.permute(1, 2, 0)) # type: ignore

    ax.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            color=color,
            linewidth=3
        )
    )

    text = f"{CLASSES[pred_class]}: {top_score:.2f}"

    ax.text(
        xmin,
        ymin,
        text,
        fontsize=12,
        color="white",
        bbox=dict(facecolor=color, edgecolor="none", alpha=0.6)
    )

    ax.set_title("Result")
    ax.axis("off")

    # ✅ popup window
    plt.show()

else:
    print("Tidak ditemukan prediksi huruf A yang valid.")