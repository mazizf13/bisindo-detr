# INI COBA LAGI
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from matplotlib.patches import Rectangle

# -----------------------------
# Path dataset & folder output
# -----------------------------
DATASET_PATH = "data/asli/train"
IMG_PATH = os.path.join(DATASET_PATH, "images")
LABEL_PATH = os.path.join(DATASET_PATH, "labels")
SAVE_DIR = "visual/pre16"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# Ambil 1 gambar contoh
# -----------------------------
img_name = sorted(os.listdir(IMG_PATH))[0]
img_path = os.path.join(IMG_PATH, img_name)
img = np.array(Image.open(img_path).convert("RGB"))

# -----------------------------
# Ambil bbox dari label YOLO
# -----------------------------
label_file = os.path.join(LABEL_PATH, os.path.splitext(img_name)[0] + ".txt")
boxes, classes = [], []

if os.path.exists(label_file):
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                classes.append(int(parts[0]))
                boxes.append([float(x) for x in parts[1:]])  # x_center, y_center, w, h

boxes = np.array(boxes) if len(boxes) > 0 else np.zeros((0,4))
classes = np.array(classes) if len(classes) > 0 else np.array([])

# -----------------------------
# Define transform per tahap
# -----------------------------
resize_500 = A.Compose(
    [A.Resize(500, 500)],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)

resize_224 = A.Compose(
    [A.Resize(224, 224)],
    bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
)

normalize_tensor = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# -----------------------------
# Apply transform
# -----------------------------
img_500 = resize_500(image=img, bboxes=boxes, class_labels=classes)
img_500_img = img_500["image"]
boxes_500 = np.array(img_500["bboxes"])
classes_500 = np.array(img_500["class_labels"])

img_224 = resize_224(image=img_500_img, bboxes=boxes_500, class_labels=classes_500)
img_224_img = img_224["image"]
boxes_224 = np.array(img_224["bboxes"])
classes_224 = np.array(img_224["class_labels"])

img_norm = normalize_tensor(image=img_224_img)["image"]
img_norm_np = img_norm.permute(1,2,0).numpy()  # H x W x C

# -----------------------------
# Fungsi untuk simpan gambar dengan bbox (angka → huruf)
# -----------------------------
def save_image_with_bbox(im, bboxes, classes, path, title=None, vmin=None, vmax=None, cmap=None):
    plt.figure(figsize=(im.shape[1]/100, im.shape[0]/100))
    plt.imshow(im, vmin=vmin, vmax=vmax, cmap=cmap)
    ax = plt.gca()

    # Mapping angka → huruf A-Z
    num_to_letter = {i: chr(ord('A') + i) for i in range(26)}

    for cls, bbox in zip(classes, bboxes):
        x_center, y_center, w, h = bbox
        xmin = (x_center - w/2) * im.shape[1]
        xmax = (x_center + w/2) * im.shape[1]
        ymin = (y_center - h/2) * im.shape[0]
        ymax = (y_center + h/2) * im.shape[0]

        rect = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                         linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # Tampilkan huruf
        label_text = num_to_letter.get(int(cls), str(int(cls)))
        ax.text(xmin, ymin, label_text, fontsize=8,
                bbox=dict(facecolor='yellow', alpha=0.5))

    if title:
        ax.set_title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

# -----------------------------
# Simpan tiap tahap
# -----------------------------
save_image_with_bbox(img, boxes, classes,
                     os.path.join(SAVE_DIR, "raw_480x640.png"), title="Raw Image")

save_image_with_bbox(img_500_img, boxes_500, classes_500,
                     os.path.join(SAVE_DIR, "resize_500x500.png"), title="Resize 500x500")

save_image_with_bbox(img_224_img, boxes_224, classes_224,
                     os.path.join(SAVE_DIR, "resize_224x224.png"), title="Resize 224x224")

# -----------------------------
# Normalized → de-normalize ke 0-1
# -----------------------------
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
img_norm_vis = (img_norm_np * std + mean)
img_norm_vis = np.clip(img_norm_vis, 0, 1)

save_image_with_bbox(img_norm_vis, boxes_224, classes_224,
                     os.path.join(SAVE_DIR, "normalized_tensor.png"),
                     title="Normalized De-normalized")

# -----------------------------
# Normalized → nilai tensor asli (bisa negatif) dengan colormap
# -----------------------------
save_image_with_bbox(img_norm_np, boxes_224, classes_224,
                     os.path.join(SAVE_DIR, "normalized_tensor_raw.png"),
                     title="Normalized Raw Tensor", vmin=-3, vmax=3, cmap='seismic')

# -----------------------------
# Print ukuran
# -----------------------------
print("Raw Image:", img.shape)
print("Resize 500x500:", img_500_img.shape)
print("Resize 224x224:", img_224_img.shape)
print("Normalized Tensor:", img_norm_np.shape)
print(f"Semua visualisasi tersimpan di folder: {SAVE_DIR}")


# INI bisa kok
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# # -----------------------------
# # Path dataset & folder output
# # -----------------------------
# DATASET_PATH = "data/asli/train"
# IMG_PATH = os.path.join(DATASET_PATH, "images")
# SAVE_DIR = "visual/pre"

# os.makedirs(SAVE_DIR, exist_ok=True)

# # Ambil 1 gambar contoh
# img_name = sorted(os.listdir(IMG_PATH))[0]
# img_path = os.path.join(IMG_PATH, img_name)
# img = np.array(Image.open(img_path).convert("RGB"))

# # -----------------------------
# # Define transform per tahap
# # -----------------------------
# resize_500 = A.Resize(500, 500)
# resize_224 = A.Resize(224, 224)

# normalize_tensor = A.Compose([
#     A.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]),
#     ToTensorV2()
# ])

# # -----------------------------
# # Apply transform
# # -----------------------------
# img_500 = resize_500(image=img)["image"]
# img_224 = resize_224(image=img_500)["image"]
# img_norm = normalize_tensor(image=img_224)["image"]  # C x H x W

# # -----------------------------
# # Convert tensor untuk visualisasi langsung (tanpa denormalisasi)
# # -----------------------------
# img_norm_np = img_norm.permute(1, 2, 0).numpy()  # H x W x C

# # -----------------------------
# # Fungsi untuk simpan gambar
# # -----------------------------
# def save_image(im, path):
#     plt.figure(figsize=(im.shape[1]/100, im.shape[0]/100))
#     plt.imshow(im)  # tampilkan langsung tensor normalisasi
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(path, bbox_inches='tight', pad_inches=0)
#     plt.close()

# # -----------------------------
# # Simpan setiap tahap
# # -----------------------------
# save_image(img, os.path.join(SAVE_DIR, "raw_480x6401.png"))
# save_image(img_500, os.path.join(SAVE_DIR, "resize_500x5001.png"))
# save_image(img_224, os.path.join(SAVE_DIR, "resize_224x2241.png"))
# save_image(img_norm_np, os.path.join(SAVE_DIR, "normalized_tensor1.png"))

# # -----------------------------
# # Print ukuran tiap tahap
# # -----------------------------
# print("Raw Image:", img.shape)
# print("Resize 500x500:", img_500.shape)
# print("Resize 224x224:", img_224.shape)
# print("Normalized Tensor (tanpa denormalisasi):", img_norm_np.shape)
# print(f"Semua visualisasi tersimpan di folder: {SAVE_DIR}")



# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# # -----------------------------
# # Path dataset & folder output
# # -----------------------------
# DATASET_PATH = "data/rgb"
# IMG_PATH = os.path.join(DATASET_PATH, "images")
# SAVE_DIR = "visual/pre"

# os.makedirs(SAVE_DIR, exist_ok=True)

# # Ambil 1 gambar contoh
# img_name = sorted(os.listdir(IMG_PATH))[0]
# img_path = os.path.join(IMG_PATH, img_name)
# img = np.array(Image.open(img_path).convert("RGB"))

# # -----------------------------
# # Define transform per tahap
# # -----------------------------
# resize_500 = A.Resize(500, 500)
# resize_224 = A.Resize(224, 224)

# normalize_tensor = A.Compose([
#     A.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]),
#     ToTensorV2()
# ])

# # -----------------------------
# # Apply transform
# # -----------------------------
# img_500 = resize_500(image=img)["image"]
# img_224 = resize_224(image=img_500)["image"]
# img_norm = normalize_tensor(image=img_224)["image"]  # C x H x W

# # -----------------------------
# # Convert tensor untuk visualisasi
# # -----------------------------
# img_norm_np = img_norm.permute(1, 2, 0).numpy()  # H x W x C

# # -----------------------------
# # Fungsi untuk simpan gambar
# # -----------------------------
# def save_image(im, path, channel=None, cmap=None):
#     plt.figure(figsize=(im.shape[1]/100, im.shape[0]/100))
#     if channel is not None:
#         plt.imshow(im[:, :, channel], cmap=cmap)
#     else:
#         plt.imshow(im)
#     plt.axis('off')
#     plt.tight_layout()
#     plt.savefig(path, bbox_inches='tight', pad_inches=0)
#     plt.close()

# # -----------------------------
# # Simpan setiap tahap
# # -----------------------------
# save_image(img, os.path.join(SAVE_DIR, "raw_480x640rgb.png"))
# save_image(img_500, os.path.join(SAVE_DIR, "resize_500x500rgb.png"))
# save_image(img_224, os.path.join(SAVE_DIR, "resize_224x224rgb.png"))
# save_image(img_norm_np, os.path.join(SAVE_DIR, "normalized_tensor_Rrgb.png"),
#            channel=0, cmap='seismic')

# # -----------------------------
# # Print ukuran tiap tahap
# # -----------------------------
# print("Raw Image:", img.shape)
# print("Resize 500x500:", img_500.shape)
# print("Resize 224x224:", img_224.shape)
# print("Normalized Tensor:", img_norm.shape)
# print(f"Semua visualisasi tersimpan di folder: {SAVE_DIR}")

# # -----------------------------
# # Print statistik nilai tensor normalisasi
# # -----------------------------
# for i, ch_name in enumerate(["R", "G", "B"]):
#     ch = img_norm[i].numpy()
#     print(f"Channel {ch_name}: min={ch.min():.3f}, max={ch.max():.3f}, mean={ch.mean():.3f}, std={ch.std():.3f}")


# # ADA DENORMALISASI NORMALIZED TENSOR
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# # -----------------------------
# # Path dataset
# # -----------------------------
# DATASET_PATH = "data/asli/train"
# IMG_PATH = os.path.join(DATASET_PATH, "images")

# # Ambil 1 gambar contoh
# img_name = sorted(os.listdir(IMG_PATH))[0]
# img_path = os.path.join(IMG_PATH, img_name)

# img = np.array(Image.open(img_path).convert("RGB"))

# # -----------------------------
# # Define transform per tahap
# # -----------------------------
# resize_500 = A.Resize(500, 500)
# resize_224 = A.Resize(224, 224)

# normalize_tensor = A.Compose([
#     A.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]),
#     ToTensorV2()
# ])

# # -----------------------------
# # Apply transform
# # -----------------------------
# img_500 = resize_500(image=img)["image"]
# img_224 = resize_224(image=img_500)["image"]

# img_norm = normalize_tensor(image=img_224)["image"]

# # Untuk visualisasi, kembalikan ke range 0-1
# img_norm_vis = img_norm.permute(1, 2, 0).numpy()
# img_norm_vis = (img_norm_vis - img_norm_vis.min()) / (img_norm_vis.max() - img_norm_vis.min())

# # -----------------------------
# # Plot hasil di figure terpisah agar proporsional
# # -----------------------------
# images = [img, img_500, img_224, img_norm_vis]
# titles = ["Raw Image 480x640", "Resize 500x500", "Resize 224x224", "Normalized Tensor"]

# for im, title in zip(images, titles):
#     plt.figure(figsize=(im.shape[1]/100, im.shape[0]/100))  # width = piksel / 100, height = piksel /100
#     plt.imshow(im)
#     plt.title(title)
#     plt.axis('off')
#     plt.show()

# # -----------------------------
# # Print ukuran asli tiap tahap
# # -----------------------------
# print("Raw Image:", img.shape)
# print("Resize 500x500:", img_500.shape)
# print("Resize 224x224:", img_224.shape)
# print("Normalized Tensor:", img_norm.shape)
