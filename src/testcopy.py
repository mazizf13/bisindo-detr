# ========== AZXX FIXED VERSION DENORMALIZED==========
import os
import torch
import time
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from data import DETRData
from model import DETR
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
from utils.logger import get_logger
from utils.rich_handlers import TestHandler, DetectionHandler

# =========================
# LOGGER
# =========================
logger = get_logger("test")
test_handler = TestHandler()
detection_handler = DetectionHandler()

logger.print_banner()

# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.test(f"Running on device: {device}")

# =========================
# DATASET
# =========================
test_dataset = DETRData("data/mei/testing", train=False)

# =========================
# MODEL
# =========================
num_classes = 43
model = DETR(num_classes=num_classes).to(device)
model.eval()

model.load_pretrained(
    "pretrained/mei/skenario2.pt",
    map_location=device
)

logger.success("Model loaded successfully")

# =========================
# CLASSES & COLORS
# =========================
CLASSES = get_classes()
COLORS = get_colors()

# =========================
# SAVE DIR
# =========================
save_dir = "test_predictions/mei/skenario2/denormalized"
os.makedirs(save_dir, exist_ok=True)

# =========================
# DENORMALIZATION FUNCTION
# =========================
def denormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1)

    return img

# =========================
# TARGET CLASSES
# =========================
TARGET_CLASSES = list(range(43))
class_samples = {}

# =========================
# PICK 1 IMAGE PER CLASS
# =========================
for i in range(len(test_dataset)):

    img, ann = test_dataset[i]
    labels = set(map(int, ann["labels"]))

    for cls in TARGET_CLASSES:
        if cls in labels and cls not in class_samples:
            class_samples[cls] = (img, ann)

    if len(class_samples) == 43:
        break

# =========================
# FILL MISSING CLASSES
# =========================
fallback_img, fallback_ann = test_dataset[0]

for cls in TARGET_CLASSES:
    if cls not in class_samples:
        class_samples[cls] = (fallback_img, fallback_ann)

class_samples = dict(sorted(class_samples.items()))

print(f"✅ Total classes prepared: {len(class_samples)} / 43")

# =========================
# INFERENCE
# =========================
with torch.no_grad():

    for cls, (img, ann) in class_samples.items():

        X = img.unsqueeze(0).to(device)

        start_time = time.time()
        result = model(X)
        inference_time = (time.time() - start_time) * 1000

        probabilities = result["pred_logits"].softmax(-1)[0, :, :-1]
        max_probs, max_classes = probabilities.max(-1)

        top_score, top_idx = max_probs.max(0)

        # =========================
        # PLOT
        # =========================
        fig, ax = plt.subplots(figsize=(4, 4))

        # 🔥 DENORMALIZED IMAGE (FIX OUTPUT COLOR)
        vis_img = denormalize(img)
        ax.imshow(vis_img.permute(1, 2, 0))

        # =========================
        # DRAW DETECTION
        # =========================
        if top_score > 0.7:

            pred_box = result["pred_boxes"][0, top_idx].unsqueeze(0)
            bbox = rescale_bboxes(pred_box, (224, 224))[0]

            pred_class = int(max_classes[top_idx])
            score = float(top_score)

            xmin, ymin, xmax, ymax = bbox.detach().cpu().numpy()

            color = tuple(float(c) / 255 for c in COLORS[pred_class])

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

            text = f"{CLASSES[pred_class]}: {score:.2f}"

            ax.text(
                xmin,
                ymin,
                text,
                fontsize=12,
                color="white",
                bbox=dict(facecolor=color, edgecolor="none", alpha=0.6)
            )

        ax.set_title(f"GT: {CLASSES[cls]}")
        ax.axis("off")

        save_path = os.path.join(save_dir, f"{CLASSES[cls]}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        detection_handler.log_inference_time(inference_time)

print(f"\n✅ 43 stable prediction images saved to: {save_dir}")


# # ========== AZXX FIXED VERSION ==========
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
# # LOGGER
# # =========================
# logger = get_logger("test")
# test_handler = TestHandler()
# detection_handler = DetectionHandler()

# logger.print_banner()

# # =========================
# # DEVICE
# # =========================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# logger.test(f"Running on device: {device}")

# # =========================
# # DATASET
# # =========================
# test_dataset = DETRData("data/mei/testing", train=False)

# # =========================
# # MODEL
# # =========================
# num_classes = 43
# model = DETR(num_classes=num_classes).to(device)
# model.eval()

# model.load_pretrained(
#     "pretrained/mei/skenario1noaug.pt",
#     map_location=device
# )

# logger.success("Model loaded successfully")

# # =========================
# # CLASSES & COLORS
# # =========================
# CLASSES = get_classes()
# COLORS = get_colors()

# # =========================
# # SAVE DIR
# # =========================
# save_dir = "test_predictions/mei/skenario1"
# os.makedirs(save_dir, exist_ok=True)

# # =========================
# # TARGET CLASSES (FIXED ORDER)
# # =========================
# TARGET_CLASSES = list(range(43))  # 0–42

# class_samples = {}

# # =========================
# # PICK 1 IMAGE PER CLASS (DETERMINISTIC)
# # =========================
# for i in range(len(test_dataset)):

#     img, ann = test_dataset[i]
#     labels = set(map(int, ann["labels"]))

#     for cls in TARGET_CLASSES:
#         if cls in labels and cls not in class_samples:
#             class_samples[cls] = (img, ann)

#     if len(class_samples) == 43:
#         break

# # =========================
# # FILL MISSING CLASSES (FALLBACK)
# # =========================
# fallback_img, fallback_ann = test_dataset[0]

# for cls in TARGET_CLASSES:
#     if cls not in class_samples:
#         class_samples[cls] = (fallback_img, fallback_ann)

# # =========================
# # FINAL CHECK
# # =========================
# class_samples = dict(sorted(class_samples.items()))

# print(f"✅ Total classes prepared: {len(class_samples)} / 43")

# # =========================
# # INFERENCE
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

# print(f"\n✅ 43 stable prediction images saved to: {save_dir}")



# # ========== AMBIL HURUF A AJA ==========
# # import torch
# # import time
# # from matplotlib import pyplot as plt
# # from matplotlib.patches import Rectangle

# # from data import DETRData
# # from model import DETR
# # from utils.boxes import rescale_bboxes
# # from utils.setup import get_classes, get_colors

# # # =========================
# # # Device
# # # =========================
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print(f"Running on device: {device}")

# # # =========================
# # # Dataset
# # # =========================
# # test_dataset = DETRData("data/mei/testing", train=False)

# # # =========================
# # # Model
# # # =========================
# # model = DETR(num_classes=43).to(device)
# # model.eval()

# # model.load_pretrained(
# #     # "pretrained/warnet2/noaug/300_model.pt",
# #     "pretrained/warnet2/aug-new/300_model.pt",
# #     map_location=device
# # )

# # print("Model loaded successfully")

# # CLASSES = get_classes()
# # COLORS = get_colors()

# # # =========================
# # # Cari SEMUA sampel huruf A
# # # =========================
# # target_class = 0  # A biasanya index 0
# # best_sample = None
# # best_score_global = 0
# # best_result = None
# # best_img = None

# # # =========================
# # # Loop semua data
# # # =========================
# # with torch.no_grad():
# #     for img, ann in test_dataset:

# #         labels = ann["labels"]

# #         # cek apakah ada huruf A
# #         if target_class not in labels:
# #             continue

# #         X = img.unsqueeze(0).to(device)

# #         start_time = time.time()
# #         result = model(X)
# #         inference_time = (time.time() - start_time) * 1000

# #         probabilities = result["pred_logits"].softmax(-1)[0, :, :-1]
# #         max_probs, max_classes = probabilities.max(-1)

# #         top_score, top_idx = max_probs.max(0)
# #         pred_class = int(max_classes[top_idx])

# #         # hanya ambil prediksi A
# #         if pred_class == target_class and top_score > best_score_global:
# #             best_score_global = float(top_score)
# #             best_sample = (img, ann)
# #             best_result = result
# #             best_img = img

# # # =========================
# # # Tampilkan hasil terbaik
# # # =========================
# # if best_sample is not None:

# #     print("\n=== RESULT ===")
# #     print(f"Confidence Score: {best_score_global:.4f}")

# #     img = best_img
# #     result = best_result

# #     probabilities = result["pred_logits"].softmax(-1)[0, :, :-1] # type: ignore
# #     max_probs, max_classes = probabilities.max(-1)
# #     top_score, top_idx = max_probs.max(0)

# #     pred_box = result["pred_boxes"][0, top_idx].unsqueeze(0) # type: ignore
# #     bbox = rescale_bboxes(pred_box, (224, 224))[0]

# #     pred_class = int(max_classes[top_idx])

# #     xmin, ymin, xmax, ymax = bbox.detach().cpu().numpy()

# #     color = tuple(float(c) / 255 for c in COLORS[pred_class])

# #     fig, ax = plt.subplots(figsize=(5, 5))
# #     ax.imshow(img.permute(1, 2, 0)) # type: ignore

# #     ax.add_patch(
# #         Rectangle(
# #             (xmin, ymin),
# #             xmax - xmin,
# #             ymax - ymin,
# #             fill=False,
# #             color=color,
# #             linewidth=3
# #         )
# #     )

# #     text = f"{CLASSES[pred_class]}: {top_score:.2f}"

# #     ax.text(
# #         xmin,
# #         ymin,
# #         text,
# #         fontsize=12,
# #         color="white",
# #         bbox=dict(facecolor=color, edgecolor="none", alpha=0.6)
# #     )

# #     ax.set_title("Result")
# #     ax.axis("off")

# #     # ✅ popup window
# #     plt.show()

# # else:
# #     print("Tidak ditemukan prediksi huruf A yang valid.")