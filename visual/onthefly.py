# DEBUG_ONTHEFLY 1 per 1 update randomsized
import os
import torch
import numpy as np
from PIL import Image
import albumentations as A
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from utils.setup import get_classes

# =====================================================
# Dataset 1 gambar
# =====================================================
class DETRDataSingle:
    def __init__(self, path):
        self.images_path = os.path.join(path, "images")
        self.labels_path = os.path.join(path, "labels")
        self.labels = sorted([f for f in os.listdir(self.labels_path) if f.endswith(".txt")])

        # Resize awal → 500x500
        self.preprocess500 = A.Compose(
            [A.Resize(500, 500)],
            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_file = self.labels[idx]
        image_name = os.path.splitext(label_file)[0]

        image_path = os.path.join(self.images_path, f"{image_name}.jpg")
        label_path = os.path.join(self.labels_path, label_file)

        image = np.array(Image.open(image_path).convert("RGB"))

        boxes, classes = [], []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        classes.append(int(parts[0]))
                        boxes.append([float(x) for x in parts[1:]])

        boxes = np.array(boxes) if len(boxes) > 0 else np.zeros((0,4))
        classes = np.array(classes) if len(classes) > 0 else np.array([])

        processed = self.preprocess500(
            image=image,
            bboxes=boxes,
            class_labels=classes
        )

        img500 = processed["image"]
        boxes500 = np.array(processed["bboxes"])
        classes500 = np.array(processed["class_labels"])

        return img500, boxes500, classes500, image_name

# =====================================================
# Augmentasi ON THE FLY (perbaikan)
# =====================================================
AUGS = {
    "Original": A.Compose(
        [A.Resize(224, 224)],
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
    ),

    "Crop": A.Compose(
        [
            A.RandomSizedBBoxSafeCrop(224, 224, p=1.0),
            A.Resize(224, 224)  # pastikan output 224x224
        ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
    ),

    "Flip": A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=1.0)
        ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
    ),

    "ColorJitter": A.Compose(
        [
            A.Resize(224, 224),
            A.ColorJitter(
                hue=0.5,
                brightness=0.5,
                contrast=0.5,
                saturation=0.5,
                p=1.0
            )
        ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
    )
}

# =====================================================
# Visualisasi
# =====================================================
if __name__ == "__main__":

    os.makedirs("debug_onthefly", exist_ok=True)

    dataset = DETRDataSingle("data/asli/train")
    CLASSES = get_classes()

    img500, boxes500, classes500, img_name = dataset[0]

    for aug_name, aug in AUGS.items():

        augmented = aug(
            image=img500,
            bboxes=boxes500,
            class_labels=classes500
        )

        img_aug = augmented["image"]
        boxes_aug = torch.tensor(augmented["bboxes"], dtype=torch.float32)
        labels_aug = torch.tensor(augmented["class_labels"], dtype=torch.long)

        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(img_aug)
        ax.axis("off")
        ax.set_title(aug_name)

        # Gambar bounding box
        for cls, bbox in zip(labels_aug, boxes_aug):
            x_center, y_center, w, h = bbox
            xmin = (x_center - w/2) * img_aug.shape[1]
            xmax = (x_center + w/2) * img_aug.shape[1]
            ymin = (y_center - h/2) * img_aug.shape[0]
            ymax = (y_center + h/2) * img_aug.shape[0]

            rect = Rectangle(
                (xmin, ymin),
                xmax-xmin,
                ymax-ymin,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                xmin,
                ymin,
                CLASSES[int(cls)],
                fontsize=8,
                bbox=dict(facecolor='yellow', alpha=0.5)
            )

        plt.tight_layout()
        plt.savefig(f"debug_onthefly/{img_name}_{aug_name}.png")
        plt.close()

# DEBUG_ONTHEFLY 1 per 1
# import os
# import torch
# import numpy as np
# from PIL import Image
# import albumentations as A
# from matplotlib import pyplot as plt
# from matplotlib.patches import Rectangle
# from utils.setup import get_classes

# # =====================================================
# # Dataset 1 gambar
# # =====================================================
# class DETRDataSingle:
#     def __init__(self, path):
#         self.images_path = os.path.join(path, "images")
#         self.labels_path = os.path.join(path, "labels")
#         self.labels = sorted([f for f in os.listdir(self.labels_path) if f.endswith(".txt")])

#         # Resize awal → 500x500
#         self.preprocess500 = A.Compose(
#             [A.Resize(500, 500)],
#             bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
#         )

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         label_file = self.labels[idx]
#         image_name = os.path.splitext(label_file)[0]

#         image_path = os.path.join(self.images_path, f"{image_name}.jpg")
#         label_path = os.path.join(self.labels_path, label_file)

#         image = np.array(Image.open(image_path).convert("RGB"))

#         boxes, classes = [], []
#         if os.path.exists(label_path):
#             with open(label_path, "r") as f:
#                 for line in f:
#                     parts = line.strip().split()
#                     if len(parts) == 5:
#                         classes.append(int(parts[0]))
#                         boxes.append([float(x) for x in parts[1:]])

#         boxes = np.array(boxes) if len(boxes) > 0 else np.zeros((0,4))
#         classes = np.array(classes) if len(classes) > 0 else np.array([])

#         processed = self.preprocess500(
#             image=image,
#             bboxes=boxes,
#             class_labels=classes
#         )

#         img500 = processed["image"]
#         boxes500 = np.array(processed["bboxes"])
#         classes500 = np.array(processed["class_labels"])

#         return img500, boxes500, classes500, image_name


# # =====================================================
# # Augmentasi ON THE FLY
# # =====================================================
# AUGS = {
#     "Original": A.Compose(
#         [A.Resize(224, 224)],
#         bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
#     ),

#     "Crop": A.Compose(
#         [A.RandomSizedBBoxSafeCrop(224, 224, p=1.0)],
#         bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
#     ),

#     "Flip": A.Compose(
#         [
#             A.Resize(224, 224),
#             A.HorizontalFlip(p=1.0)
#         ],
#         bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
#     ),

#     "ColorJitter": A.Compose(
#         [
#             A.Resize(224, 224),
#             A.ColorJitter(
#                 hue=0.5,
#                 brightness=0.5,
#                 contrast=0.5,
#                 saturation=0.5,
#                 p=1.0
#             )
#         ],
#         bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
#     )
# }

# # =====================================================
# # Visualisasi
# # =====================================================
# if __name__ == "__main__":

#     os.makedirs("debug_onthefly", exist_ok=True)

#     dataset = DETRDataSingle("data/asli/train")
#     CLASSES = get_classes()

#     img500, boxes500, classes500, img_name = dataset[0]

#     for aug_name, aug in AUGS.items():

#         augmented = aug(
#             image=img500,
#             bboxes=boxes500,
#             class_labels=classes500
#         )

#         img_aug = augmented["image"]
#         boxes_aug = torch.tensor(augmented["bboxes"], dtype=torch.float32)
#         labels_aug = torch.tensor(augmented["class_labels"], dtype=torch.long)

#         fig, ax = plt.subplots(figsize=(5,5))

#         ax.imshow(img_aug)
#         ax.axis("off")

#         # Title cukup nama augmentasi
#         ax.set_title(aug_name)

#         for cls, bbox in zip(labels_aug, boxes_aug):

#             x_center, y_center, w, h = bbox

#             xmin = (x_center - w/2) * img_aug.shape[1]
#             xmax = (x_center + w/2) * img_aug.shape[1]
#             ymin = (y_center - h/2) * img_aug.shape[0]
#             ymax = (y_center + h/2) * img_aug.shape[0]

#             rect = Rectangle(
#                 (xmin, ymin),
#                 xmax-xmin,
#                 ymax-ymin,
#                 linewidth=2,
#                 edgecolor='red',
#                 facecolor='none'
#             )

#             ax.add_patch(rect)

#             ax.text(
#                 xmin,
#                 ymin,
#                 CLASSES[int(cls)],
#                 fontsize=8,
#                 bbox=dict(facecolor='yellow', alpha=0.5)
#             )

#         plt.tight_layout()

#         plt.savefig(f"debug_onthefly/{img_name}_{aug_name}.png")
#         plt.close()

# # import os
# # import torch
# # import numpy as np
# # from torch.utils.data import Dataset, DataLoader
# # from PIL import Image
# # import albumentations as A
# # from albumentations.pytorch import ToTensorV2
# # from matplotlib import pyplot as plt
# # from matplotlib.patches import Rectangle

# # from utils.boxes import rescale_bboxes, stacker
# # from utils.setup import get_classes

# # # =========================================================
# # # DENORMALIZATION FUNCTION (UNTUK VISUALISASI)
# # # =========================================================
# # def denormalize(img_tensor, mean, std):
# #     mean = torch.tensor(mean).view(3, 1, 1)
# #     std = torch.tensor(std).view(3, 1, 1)
# #     img = img_tensor * std + mean
# #     return img.clamp(0, 1)

# # # =========================================================
# # # DATASET TANPA RANDOM AUGMENTASI (UNTUK VISUALISASI)
# # # =========================================================
# # class DETRData(Dataset):
# #     def __init__(self, path):
# #         super().__init__()
# #         self.path = path
# #         self.labels_path = os.path.join(path, "labels")
# #         self.images_path = os.path.join(path, "images")

# #         self.labels = [x for x in os.listdir(self.labels_path) if x.endswith(".txt")]
# #         self.image_map = {os.path.splitext(f)[0]: f for f in os.listdir(self.images_path)}

# #     def __len__(self):
# #         return len(self.labels)

# #     def __getitem__(self, idx):
# #         label_name = os.path.splitext(self.labels[idx])[0]
# #         label_path = os.path.join(self.labels_path, self.labels[idx])
# #         image_path = os.path.join(self.images_path, self.image_map[label_name])

# #         img = np.array(Image.open(image_path).convert("RGB"))

# #         class_labels, bounding_boxes = [], []
# #         with open(label_path) as f:
# #             for line in f:
# #                 cls, *bbox = line.strip().split()
# #                 class_labels.append(int(cls))
# #                 bounding_boxes.append([float(x) for x in bbox])

# #         return img, np.array(bounding_boxes), np.array(class_labels)

# # # =========================================================
# # # VISUALISASI AUGMENTASI DETERMINISTIK
# # # =========================================================
# # if __name__ == "__main__":
# #     dataset = DETRData("data/asli/train")
# #     img, bboxes, labels = dataset[0]  # ambil satu gambar untuk visualisasi
# #     CLASSES = get_classes()

# #     # =========================================================
# #     # Buat transformasi deterministik (p=1.0)
# #     # =========================================================
# #     augmentations = [
# #         ("Original/NoAug", A.Compose([
# #             A.Resize(500, 500),
# #             A.Resize(224, 224),
# #             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# #             ToTensorV2()
# #         ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))),

# #         ("HorizontalFlip", A.Compose([
# #             A.Resize(500, 500),
# #             A.Resize(224, 224),
# #             A.HorizontalFlip(p=1.0),
# #             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# #             ToTensorV2()
# #         ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))),

# #         ("ColorJitter", A.Compose([
# #             A.Resize(500, 500),
# #             A.Resize(224, 224),
# #             A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.15, hue=0.5, p=1.0),
# #             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# #             ToTensorV2()
# #         ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))),

# #         ("HueSatVal+BrightCont", A.Compose([
# #             A.Resize(500, 500),
# #             A.Resize(224, 224),
# #             A.HueSaturationValue(hue_shift_limit=60, sat_shift_limit=40, val_shift_limit=15, p=1.0),
# #             A.RandomBrightnessContrast(brightness_limit=(-0.1,0.25), contrast_limit=(-0.1,0.25), p=1.0),
# #             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# #             ToTensorV2()
# #         ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))),

# #         ("RandomCrop", A.Compose([
# #             A.Resize(500, 500),
# #             A.RandomSizedBBoxSafeCrop(height=224, width=224, erosion_rate=0.0, p=1.0),
# #             A.Resize(224, 224),
# #             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# #             ToTensorV2()
# #         ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))),

# #         ("Flip+ColorJitter", A.Compose([
# #             A.Resize(500, 500),
# #             A.Resize(224, 224),
# #             A.HorizontalFlip(p=1.0),
# #             A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.15, hue=0.5, p=1.0),
# #             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# #             ToTensorV2()
# #         ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))),

# #         ("Flip+HueSatVal", A.Compose([
# #             A.Resize(500, 500),
# #             A.Resize(224, 224),
# #             A.HorizontalFlip(p=1.0),
# #             A.HueSaturationValue(hue_shift_limit=60, sat_shift_limit=40, val_shift_limit=15, p=1.0),
# #             A.RandomBrightnessContrast(brightness_limit=(-0.1,0.25), contrast_limit=(-0.1,0.25), p=1.0),
# #             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# #             ToTensorV2()
# #         ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))),
# #         ("Flip+HueSatVal+RandomCrop", A.Compose([
# #             A.Resize(500, 500),
# #             A.Resize(224, 224),
# #             A.HorizontalFlip(p=1.0),
# #             A.HueSaturationValue(hue_shift_limit=60, sat_shift_limit=40, val_shift_limit=15, p=1.0),
# #             A.RandomBrightnessContrast(brightness_limit=(-0.1,0.25), contrast_limit=(-0.1,0.25), p=1.0),
# #             A.RandomSizedBBoxSafeCrop(height=224, width=224, erosion_rate=0.0, p=1.0),
# #             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# #             ToTensorV2()
# #         ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"])))
# #     ]

# #     # =========================================================
# #     # Visualisasi
# #     # =========================================================
# #     fig, ax = plt.subplots(2, 4, figsize=(16, 8))
# #     axs = ax.flatten()

# #     for i, (title, aug) in enumerate(augmentations):
# #         out = aug(image=img, bboxes=bboxes, class_labels=labels)
# #         img_aug = denormalize(out["image"], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# #         ax_ = axs[i]
# #         ax_.imshow(img_aug.permute(1, 2, 0))
# #         ax_.axis("off")
# #         ax_.set_title(title, fontsize=12)

# #         boxes = rescale_bboxes(torch.tensor(out["bboxes"], dtype=torch.float32), (224,224))
# #         for cls, bbox in zip(out["class_labels"], boxes):
# #             xmin, ymin, xmax, ymax = bbox.numpy()
# #             ax_.add_patch(Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
# #                                     linewidth=2, edgecolor="red", facecolor="none"))
# #             ax_.text(xmin, ymin, CLASSES[cls], fontsize=10, bbox=dict(facecolor="yellow", alpha=0.6))

# #     plt.tight_layout()
# #     plt.show()

# # import torch
# # import numpy as np
# # from torch.utils.data import DataLoader, Dataset 
# # import os 
# # from PIL import Image 
# # import albumentations as A
# # import numpy as np
# # from colorama import Fore 
# # from matplotlib import pyplot as plt 
# # from utils.boxes import rescale_bboxes, stacker
# # from utils.setup import get_classes
# # from utils.logger import get_logger
# # from utils.rich_handlers import DataLoaderHandler
# # import sys 


# # class DETRData(Dataset): 
# #     def __init__(self, path, train=True):
# #         super().__init__()
# #         self.path = path
# #         self.labels_path = os.path.join(self.path, 'labels')
# #         self.images_path = os.path.join(self.path, 'images')
# #         self.label_files = os.listdir(self.labels_path) 
# #         self.labels = list(filter(lambda x: x.endswith('.txt'), self.label_files))
# #         self.train = train
        
# #         # Initialize logger
# #         self.logger = get_logger("data_loader")
# #         self.data_handler = DataLoaderHandler()
        
# #         # Log dataset initialization
# #         dataset_info = {
# #             "Dataset Path": self.path,
# #             "Mode": "Training" if train else "Testing",
# #             "Total Samples": len(self.labels),
# #             "Images Path": self.images_path,
# #             "Labels Path": self.labels_path
# #         }
# #         self.data_handler.log_dataset_stats(dataset_info)
        
# #         # Log transforms information
# #         transform_list = [
# #             "Resize to 500x500",
# #             "Random Crop 224x224 (training only)",
# #             "Final Resize to 224x224",
# #             "Horizontal Flip p=0.5 (training only)",
# #             "Color Jitter (training only)",
# #             "Normalize (ImageNet stats)",
# #             "Convert to Tensor"
# #         ]
# #         self.data_handler.log_transform_info(transform_list)             

# #     def safe_transform(self, image, bboxes, labels, max_attempts=50):
# #         self.transform = A.Compose(
# #             [   
# #                 A.Resize(500,500),
# #                 *([A.RandomSizedBBoxSafeCrop(width=224, height=224, p=0.33)] if self.train else []), # Example random crop
# #                 A.Resize(224,224),
# #                 *([A.HorizontalFlip(p=0.5)] if self.train else []),
# #                 *([A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.5, p=0.5)] if self.train else []),
# #                 A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# #                 A.ToTensorV2()
# #             ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
# #         )
        
# #         for attempt in range(max_attempts):
# #             try:
# #                 transformed = self.transform(image=image, bboxes=bboxes, class_labels=labels)
# #                 # Check if we still have bboxes after transformation
# #                 if len(transformed['bboxes']) > 0:
# #                     return transformed
# #             except:
# #                 continue
        
# #         return {'image': image, 'bboxes': bboxes, 'class_labels': labels}

# #     def __len__(self): 
# #         return len(self.labels) 

# #     def __getitem__(self, idx): 
# #         self.label_path = os.path.join(self.labels_path, self.labels[idx]) 
# #         self.image_name = self.labels[idx].split('.')[0]
# #         self.image_path = os.path.join(self.images_path, f'{self.image_name}.jpg') 
        
# #         img = Image.open(self.image_path)
# #         with open(self.label_path, 'r') as f: 
# #             annotations = f.readlines()
# #         class_labels = []
# #         bounding_boxes = []
# #         for annotation in annotations: 
# #             annotation = annotation.split('\n')[:-1][0].split(' ')
# #             class_labels.append(annotation[0]) 
# #             bounding_boxes.append(annotation[1:])
# #         class_labels = np.array(class_labels).astype(int) 
# #         bounding_boxes = np.array(bounding_boxes).astype(float) 

# #         augmented = self.safe_transform(image=np.array(img), bboxes=bounding_boxes, labels=class_labels)
# #         augmented_img_tensor = augmented['image']
# #         augmented_bounding_boxes = np.array(augmented['bboxes'])
# #         augmented_classes = augmented['class_labels']

# #         labels = torch.tensor(augmented_classes, dtype=torch.long)  
# #         boxes = torch.tensor(augmented_bounding_boxes, dtype=torch.float32)
# #         return augmented_img_tensor, {'labels': labels, 'boxes': boxes}

# import os
# import torch
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from matplotlib import pyplot as plt
# import re 

# from utils.boxes import rescale_bboxes
# from utils.setup import get_classes

# # =====================================================
# # Custom Collate Function (karena return 3 item)
# # =====================================================
# def stacker(batch):
#     images, targets, names = zip(*batch)
#     return list(images), list(targets), list(names)


# # =====================================================
# # Dataset
# # =====================================================
# class DETRData(Dataset):
#     def __init__(self, path, train=True):
#         super().__init__()
#         self.train = train

#         self.images_path = os.path.join(path, "images")
#         self.labels_path = os.path.join(path, "labels")

#         self.labels = sorted([
#             f for f in os.listdir(self.labels_path)
#             if f.endswith(".txt")
#         ])

#         self.transform = A.ReplayCompose(
#             [
#                 A.Resize(500, 500),
#                 *([A.RandomSizedBBoxSafeCrop(224, 224, p=0.33)] if train else []),
#                 A.Resize(224, 224),
#                 *([A.HorizontalFlip(p=0.5)] if train else []),
#                 *([A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.5, p=0.5)] if train else []),
#                 A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                 ToTensorV2()
#             ],
#             bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'])
#         )

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         label_file = self.labels[idx]
#         image_name = os.path.splitext(label_file)[0]

#         image_path = os.path.join(self.images_path, f"{image_name}.jpg")
#         label_path = os.path.join(self.labels_path, label_file)

#         image = np.array(Image.open(image_path).convert("RGB"))

#         boxes, classes = [], []

#         if os.path.exists(label_path):
#             with open(label_path, "r") as f:
#                 for line in f:
#                     parts = line.strip().split()
#                     if len(parts) == 5:
#                         classes.append(int(parts[0]))
#                         boxes.append([float(x) for x in parts[1:]])

#         boxes = np.array(boxes) if len(boxes) > 0 else np.zeros((0, 4))
#         classes = np.array(classes) if len(classes) > 0 else np.array([])

#         augmented = self.transform(
#             image=image,
#             bboxes=boxes,
#             class_labels=classes
#         )

#         image_tensor = augmented["image"]
#         boxes_tensor = torch.tensor(augmented["bboxes"], dtype=torch.float32) if len(augmented["bboxes"]) > 0 else torch.zeros((0, 4), dtype=torch.float32)
#         labels_tensor = torch.tensor(augmented["class_labels"], dtype=torch.long)

#         replay = augmented["replay"]
#         applied_transforms = [t["__class_fullname__"].split(".")[-1] for t in replay["transforms"] if t["applied"]]
#         aug_name = "_".join(applied_transforms) if applied_transforms else "no_aug"
#         aug_name_clean = re.sub(r'[^a-zA-Z0-9_]', '', aug_name)

#         # gabungkan dengan nama image
#         augmented_name = f"{aug_name_clean}"

#         target = {"labels": labels_tensor, "boxes": boxes_tensor}
#         return image_tensor, target, augmented_name


# # =====================================================
# # Visualisasi Test (NORMALISASI)
# # =====================================================
# if __name__ == "__main__":
#     dataset = DETRData("data/asli/train", train=True)
#     dataloader = DataLoader(dataset, batch_size=8, collate_fn=stacker, drop_last=True)

#     X, y, names = next(iter(dataloader))
#     CLASSES = get_classes()

#     # Print nama augmentasi ke terminal
#     print("Nama augmentasi untuk batch ini:")
#     for name in names:
#         print(name)

#     fig, axes = plt.subplots(2, 4, figsize=(12, 6))
#     axes = axes.flatten()

#     for img, annotations, name, ax in zip(X, y, names, axes):
#         # langsung tampil image yang sudah dinormalisasi
#         ax.imshow(img.permute(1, 2, 0))
#         ax.set_title(name, fontsize=6)
#         ax.axis("off")

#         boxes = rescale_bboxes(annotations["boxes"], (224, 224))
#         labels = annotations["labels"]

#         for cls, bbox in zip(labels, boxes):
#             xmin, ymin, xmax, ymax = bbox.detach().numpy()
#             rect = plt.Rectangle( # type: ignore
#                 (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, linewidth=2
#             )
#             ax.add_patch(rect)
#             ax.text(
#                 xmin, ymin, CLASSES[int(cls)], fontsize=8,
#                 bbox=dict(facecolor="yellow", alpha=0.5)
#             )

#     plt.tight_layout()
#     plt.show()

