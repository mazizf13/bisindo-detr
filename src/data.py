import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from colorama import Fore
from matplotlib import pyplot as plt

from utils.boxes import rescale_bboxes, stacker
from utils.setup import get_classes
from utils.logger import get_logger
from utils.rich_handlers import DataLoaderHandler
from matplotlib.patches import Rectangle


class DETRData(Dataset):
    def __init__(self, path, train=True):
        super().__init__()
        self.path = path
        self.labels_path = os.path.join(self.path, "labels")
        self.images_path = os.path.join(self.path, "images")
        self.train = train

        self.labels = [x for x in os.listdir(self.labels_path) if x.endswith(".txt")]

        # logger
        self.logger = get_logger("data_loader")
        self.data_handler = DataLoaderHandler()

        dataset_info = {
            "Dataset Path": self.path,
            "Mode": "Training" if train else "Testing",
            "Total Samples": len(self.labels),
            "Images Path": self.images_path,
            "Labels Path": self.labels_path,
        }
        self.data_handler.log_dataset_stats(dataset_info)

        transform_list = [
            "Resize to 500x500",
            "Random Crop 224x224 (training only)",
            "Final Resize to 224x224",
            "Horizontal Flip p=0.5 (training only)",
            "Color Jitter (training only)",
            "Normalize (ImageNet stats)",
            "Convert to Tensor",
        ]
        self.data_handler.log_transform_info(transform_list)

        train_transforms = [
            A.Resize(500, 500),
            A.RandomSizedBBoxSafeCrop(
                height=224,
                width=224,
                erosion_rate=0.0,
                p=0.33,
            ),
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),

            A.OneOf(
                [
                    A.ColorJitter(
                        brightness=0.1,
                        contrast=0.1,
                        saturation=0.15,
                        hue=0.5,
                        p=1.0,
                    ),

                    A.Compose([
                        A.HueSaturationValue(
                            hue_shift_limit=60,
                            sat_shift_limit=40,
                            val_shift_limit=15,
                            p=1.0,
                        ),
                        A.RandomBrightnessContrast(
                            brightness_limit=(-0.1, 0.25),
                            contrast_limit=(-0.1, 0.25),
                            p=1.0,
                        ),
                    ]),
                ],
                p=0.66,
            ),

            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]

        test_transforms = [
            A.Resize(500, 500),
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]

        self.transform = A.Compose(
            train_transforms if self.train else test_transforms,
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )

        # --- Build image map (basename -> filename) ---
        self.image_map = {}
        for img in os.listdir(self.images_path):
            base = os.path.splitext(img)[0]
            self.image_map[base] = img

        # Debug mismatch
        label_bases = set(os.path.splitext(x)[0] for x in self.labels)
        image_bases = set(self.image_map.keys())
        missing = sorted(label_bases - image_bases)
        if len(missing) > 0:
            print(
                Fore.YELLOW
                + f"⚠️ {len(missing)} label tidak punya pasangan image. Contoh: {missing[:5]}"
                + Fore.RESET
            )

    def safe_transform(self, image, bboxes, labels, max_attempts=50):
        for _ in range(max_attempts):
            try:
                transformed = self.transform(
                    image=image, bboxes=bboxes, class_labels=labels
                )
                if len(transformed["bboxes"]) > 0:
                    return transformed
            except Exception:
                continue

        return {"image": image, "bboxes": bboxes, "class_labels": labels}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_name = os.path.splitext(self.labels[idx])[0]
        label_path = os.path.join(self.labels_path, self.labels[idx])

        # exact match
        if label_name not in self.image_map:
            # fallback: cari yang mirip (Roboflow kadang beda suffix)
            candidates = [k for k in self.image_map if k.startswith(label_name)]
            if len(candidates) == 0:
                raise RuntimeError(f"Label tanpa image: {label_name}")
            label_name = candidates[0]

        image_path = os.path.normpath(
            os.path.join(self.images_path, self.image_map[label_name])
        )

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Gagal baca gambar: {image_path}\n{e}")

        with open(label_path, "r") as f:
            annotations = f.readlines()

        class_labels, bounding_boxes = [], []
        for ann in annotations:
            parts = ann.strip().split()
            if len(parts) == 5:
                class_labels.append(int(parts[0]))
                bounding_boxes.append([float(x) for x in parts[1:]])

        class_labels = np.array(class_labels)
        bounding_boxes = np.array(bounding_boxes)

        augmented = self.safe_transform(
            image=np.array(img),
            bboxes=bounding_boxes,
            labels=class_labels,
        )

        return augmented["image"], {
            "labels": torch.tensor(augmented["class_labels"], dtype=torch.long),
            "boxes": torch.tensor(
                np.array(augmented["bboxes"]), dtype=torch.float32
            ),
        }


if __name__ == "__main__":
    # dataset = DETRData("data/lawas-test_ood", train=False)
    dataset = DETRData("data/asli/train", train=True)
    dataloader = DataLoader(
        dataset, collate_fn=stacker, batch_size=16, drop_last=True
    )

    X, y = next(iter(dataloader))
    print(Fore.LIGHTCYAN_EX + str(y) + Fore.RESET)

    CLASSES = get_classes()
    fig, ax = plt.subplots(4, 4)
    axs = ax.flatten()

    for img, annotations, ax in zip(X, y, axs):
        ax.imshow(img.permute(1, 2, 0))
        box_classes = annotations["labels"]
        boxes = rescale_bboxes(annotations["boxes"], (224, 224))

        for box_class, bbox in zip(box_classes, boxes):
            if box_class != 26:
                xmin, ymin, xmax, ymax = bbox.detach().numpy()
                ax.add_patch(
                    Rectangle(
                        (xmin, ymin),
                        xmax - xmin,
                        ymax - ymin,
                        fill=False,
                        linewidth=3,
                    )
                )
                text = f"{CLASSES[box_class]}"
                ax.text(
                    xmin,
                    ymin,
                    text,
                    fontsize=15,
                    bbox=dict(facecolor="yellow", alpha=0.5),
                )

    fig.tight_layout()
    plt.show()


# import torch
# import numpy as np
# from torch.utils.data import DataLoader, Dataset 
# import os 
# from PIL import Image 
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# import numpy as np
# from colorama import Fore 
# from matplotlib import pyplot as plt 
# from utils.boxes import rescale_bboxes, stacker
# from utils.setup import get_classes
# from utils.logger import get_logger
# from utils.rich_handlers import DataLoaderHandler
# import sys 


# class DETRData(Dataset): 
#     def __init__(self, path, train=True):
#         super().__init__()
#         self.path = path
#         self.labels_path = os.path.join(self.path, 'labels')
#         self.images_path = os.path.join(self.path, 'images')
#         self.label_files = os.listdir(self.labels_path) 
#         self.labels = list(filter(lambda x: x.endswith('.txt'), self.label_files))
#         self.train = train
        
#         # logger
#         self.logger = get_logger("data_loader")
#         self.data_handler = DataLoaderHandler()
        
#         # Log dataset initialization
#         dataset_info = {
#             "Dataset Path": self.path,
#             "Mode": "Training" if train else "Testing",
#             "Total Samples": len(self.labels),
#             "Images Path": self.images_path,
#             "Labels Path": self.labels_path
#         }
#         self.data_handler.log_dataset_stats(dataset_info)
        
#         # Log transforms information
#         transform_list = [
#             "Resize to 224x224",
#             "Horizontal Flip p=0.5 (training only)",
#             "Shift/Scale/Rotate (training only)",
#             "Color Jitter (training only)",
#             "Normalize (ImageNet stats)",
#             "Convert to Tensor"
#         ]
#         self.data_handler.log_transform_info(transform_list)
        
#         # Initialize transforms
#         self._init_transforms()

#     def _init_transforms(self):
#         """Initialize training and validation transforms"""
#         # Training transforms
#         self.train_transform = A.Compose(
#             [   
#                 A.LongestMaxSize(max_size=224),  # Resize maintaining aspect ratio
#                 A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=(0,0,0)),
#                 A.HorizontalFlip(p=0.5),
#                 A.ShiftScaleRotate(
#                     shift_limit=0.1, 
#                     scale_limit=0.15, 
#                     rotate_limit=10, 
#                     border_mode=0,
#                     p=0.5
#                 ),
#                 A.ColorJitter(
#                     brightness=0.3, 
#                     contrast=0.3, 
#                     saturation=0.3, 
#                     hue=0.1, 
#                     p=0.4
#                 ),
#                 A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                 ToTensorV2()
#             ], 
#             bbox_params=A.BboxParams(
#                 format='yolo', 
#                 label_fields=['class_labels'],
#                 min_area=16,  # Minimum area in pixels
#                 min_visibility=0.3  # Keep only if >30% visible
#             )
#         )
        
#         # Validation/Test transforms (no augmentation)
#         self.val_transform = A.Compose(
#             [   
#                 A.LongestMaxSize(max_size=224),
#                 A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=(0,0,0)),
#                 A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                 ToTensorV2()
#             ], 
#             bbox_params=A.BboxParams(
#                 format='yolo', 
#                 label_fields=['class_labels'],
#                 min_area=1,
#                 min_visibility=0.1
#             )
#         )
        
#         # Fallback transform (minimal, always works)
#         self.fallback_transform = A.Compose(
#             [   
#                 A.Resize(224, 224),
#                 A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                 ToTensorV2()
#             ], 
#             bbox_params=A.BboxParams(
#                 format='yolo', 
#                 label_fields=['class_labels']
#             )
#         )

#     def safe_transform(self, image, bboxes, labels, max_attempts=10):
#         """
#         Apply transforms with fallback mechanism.
#         Returns transformed image, bboxes, and labels.
#         """
#         # Choose transform based on mode
#         transform = self.train_transform if self.train else self.val_transform
        
#         # Filter out invalid bboxes (width or height = 0)
#         valid_indices = []
#         for i, bbox in enumerate(bboxes):
#             cx, cy, w, h = bbox
#             if w > 0 and h > 0:
#                 valid_indices.append(i)
        
#         if len(valid_indices) == 0:
#             # No valid bboxes, use fallback without bboxes
#             self.logger.warning(f"No valid bboxes found, skipping bbox transform")
#             transform_no_bbox = A.Compose([
#                 A.Resize(224, 224),
#                 A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#                 ToTensorV2()
#             ])
#             transformed = transform_no_bbox(image=image)
#             return transformed['image'], np.array([]), np.array([])
        
#         bboxes = [bboxes[i] for i in valid_indices]
#         labels = [labels[i] for i in valid_indices]
        
#         # Try main transform multiple times
#         for attempt in range(max_attempts):
#             try:
#                 transformed = transform(image=image, bboxes=bboxes, class_labels=labels)
#                 # Check if still have bboxes after transformation
#                 if len(transformed['bboxes']) > 0:
#                     return (
#                         transformed['image'],
#                         np.array(transformed['bboxes']),
#                         np.array(transformed['class_labels'])
#                     )
#             except Exception as e:
#                 if attempt == max_attempts - 1:
#                     self.logger.warning(f"Transform failed after {max_attempts} attempts: {str(e)}")
#                 continue
        
#         # If all attempts fail, use fallback transform
#         try:
#             self.logger.warning("Using fallback transform (no augmentation)")
#             transformed = self.fallback_transform(image=image, bboxes=bboxes, class_labels=labels)
#             if len(transformed['bboxes']) > 0:
#                 return (
#                     transformed['image'],
#                     np.array(transformed['bboxes']),
#                     np.array(transformed['class_labels'])
#                 )
#         except Exception as e:
#             self.logger.error(f"Fallback transform failed: {str(e)}")
        
#         # Last resort: simple resize without bbox params
#         self.logger.error("All transforms failed, using minimal transform without bboxes")
#         transform_minimal = A.Compose([
#             A.Resize(224, 224),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ToTensorV2()
#         ])
#         transformed = transform_minimal(image=image)
#         return transformed['image'], np.array([]), np.array([])

#     def __len__(self): 
#         return len(self.labels) 

#     def __getitem__(self, idx): 
#         self.label_path = os.path.join(self.labels_path, self.labels[idx]) 
#         self.image_name = self.labels[idx].split('.')[0]
#         self.image_path = os.path.join(self.images_path, f'{self.image_name}.jpg') 
        
#         # Read image
#         try:
#             img = Image.open(self.image_path).convert('RGB')
#         except Exception as e:
#             self.logger.error(f"Error loading image {self.image_path}: {str(e)}")
#             # Return dummy data
#             dummy_img = torch.zeros(3, 224, 224)
#             dummy_labels = torch.tensor([], dtype=torch.long)
#             dummy_boxes = torch.tensor([], dtype=torch.float32).reshape(0, 4)
#             return dummy_img, {'labels': dummy_labels, 'boxes': dummy_boxes}
        
#         # Read labels
#         try:
#             with open(self.label_path, 'r') as f: 
#                 annotations = f.readlines()
            
#             if len(annotations) == 0:
#                 # Empty label file
#                 class_labels = np.array([])
#                 bounding_boxes = np.array([]).reshape(0, 4)
#             else:
#                 class_labels = []
#                 bounding_boxes = []
#                 for annotation in annotations:
#                     annotation = annotation.strip()
#                     if not annotation:  # Skip empty lines
#                         continue
#                     parts = annotation.split(' ')
#                     if len(parts) != 5:  # Should be: class cx cy w h
#                         self.logger.warning(f"Invalid annotation in {self.label_path}: {annotation}")
#                         continue
#                     class_labels.append(int(parts[0])) 
#                     bounding_boxes.append([float(x) for x in parts[1:]])
                
#                 class_labels = np.array(class_labels).astype(int) 
#                 bounding_boxes = np.array(bounding_boxes).astype(float)
                
#         except Exception as e:
#             self.logger.error(f"Error loading labels {self.label_path}: {str(e)}")
#             class_labels = np.array([])
#             bounding_boxes = np.array([]).reshape(0, 4)

#         # Apply transforms
#         augmented_img_tensor, augmented_bounding_boxes, augmented_classes = self.safe_transform(
#             image=np.array(img), 
#             bboxes=bounding_boxes, 
#             labels=class_labels
#         )

#         # Convert to torch tensors
#         labels = torch.tensor(augmented_classes, dtype=torch.long)  
#         boxes = torch.tensor(augmented_bounding_boxes, dtype=torch.float32)

#         # Store also original image before augmentation (just resized for consistency)
#         orig_image = np.array(img.resize((224, 224)))  # keep shape similar to augmented one
#         orig_image_tensor = torch.tensor(orig_image / 255.0).permute(2, 0, 1)  # to tensor format [C,H,W]

#         return {
#             "orig_img": orig_image_tensor,
#             "aug_img": augmented_img_tensor,
#             "labels": labels,
#             "boxes": boxes
#         }
        
#         # Ensure boxes have correct shape even if empty
#         if len(boxes) == 0:
#             boxes = boxes.reshape(0, 4)
        
#         return augmented_img_tensor, {'labels': labels, 'boxes': boxes}


# if __name__ == '__main__':
#     # Test the dataset
#     dataset = DETRData('data/train', train=True)
#     dataloader = DataLoader(dataset, collate_fn=stacker, batch_size=4, drop_last=True)

#     X, y = next(iter(dataloader))
#     print(Fore.LIGHTCYAN_EX + "Batch loaded successfully!" + Fore.RESET)
#     print(f"Images shape: {X.shape}")
#     print(f"Number of samples with annotations: {sum([len(ann['labels']) > 0 for ann in y])}")
#     batch = next(iter(dataloader))
#     orig_imgs = batch["orig_img"]
#     aug_imgs = batch["aug_img"]
#     anns = [{"labels": batch["labels"], "boxes": batch["boxes"]} for _ in range(4)]

#     fig, axes = plt.subplots(4, 3, figsize=(14, 14))
#     for i in range(4):
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

#         # ---- 1️⃣ Original Image ----
#         axes[i, 0].imshow(orig_imgs[i].permute(1,2,0))
#         axes[i, 0].set_title(f"Original Image {i+1}")
#         axes[i, 0].axis('off')

#         # ---- 2️⃣ Augmented Image ----
#         img_denorm = aug_imgs[i] * std + mean
#         img_denorm = torch.clamp(img_denorm, 0, 1)
#         axes[i, 1].imshow(img_denorm.permute(1,2,0))
#         axes[i, 1].set_title("After Augmentation")
#         axes[i, 1].axis('off')

#         # ---- 3️⃣ Tensor Values ----
#         tensor_vals = aug_imgs[i].flatten()[:12].numpy()  # sample few values
#         axes[i, 2].text(0.1, 0.5, f"Tensor sample:\n{tensor_vals}", fontsize=9)
#         axes[i, 2].set_xticks([])
#         axes[i, 2].set_yticks([])
#         axes[i, 2].set_title("Tensor Preview")

#     plt.tight_layout()
#     plt.savefig('dataset_comparison.png', dpi=150)
#     plt.show()
    
#     CLASSES = get_classes() 
#     fig, ax = plt.subplots(2, 2, figsize=(12, 12)) 
#     axs = ax.flatten()
    
#     for idx, (img, annotations, ax) in enumerate(zip(X, y, axs)): 
#         # Denormalize for visualization
#         mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
#         std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
#         img_denorm = img * std + mean
#         img_denorm = torch.clamp(img_denorm, 0, 1)
        
#         ax.imshow(img_denorm.permute(1, 2, 0))
#         ax.set_title(f"Sample {idx+1}: {len(annotations['labels'])} objects")
        
#         if len(annotations['labels']) > 0:
#             box_classes = annotations['labels'] 
#             boxes = rescale_bboxes(annotations['boxes'], (224, 224))
            
#             for box_class, bbox in zip(box_classes, boxes): 
#                 if box_class < len(CLASSES):  # Valid class
#                     xmin, ymin, xmax, ymax = bbox.detach().numpy()
#                     ax.add_patch(plt.Rectangle(
#                         (xmin, ymin), 
#                         xmax - xmin, 
#                         ymax - ymin, 
#                         fill=False, 
#                         color=(0.000, 0.447, 0.741), 
#                         linewidth=2
#                     ))
#                     text = f'{CLASSES[box_class]}'
#                     ax.text(
#                         xmin, ymin-2, text, 
#                         fontsize=10, 
#                         bbox=dict(facecolor='yellow', alpha=0.7, pad=2)
#                     )
#         ax.axis('off')

#     fig.tight_layout() 
#     plt.savefig('dataset_visualization.png', dpi=150, bbox_inches='tight')
#     print(f"{Fore.GREEN}Visualization saved to dataset_visualization.png{Fore.RESET}")
#     plt.show()