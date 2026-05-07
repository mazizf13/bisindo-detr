import cv2
import random
import torch
import numpy as np


class ManualTransform:

    def __init__(self, train=True):
        self.train = train

    def __call__(self, image, bboxes, class_labels):

        # =====================================
        # 1. RESIZE 500x500
        # =====================================

        image = cv2.resize(image, (500, 500))

        crop_size = 224

        # =====================================
        # 2. RANDOM CROP
        # =====================================

        if self.train and random.random() < 0.33:

            max_x = 500 - crop_size
            max_y = 500 - crop_size

            crop_x = random.randint(0, max_x)
            crop_y = random.randint(0, max_y)

            cropped = image[
                crop_y:crop_y + crop_size,
                crop_x:crop_x + crop_size
            ]

            new_boxes = []
            new_labels = []

            for bbox, label in zip(bboxes, class_labels):

                x_center, y_center, bw, bh = bbox

                # YOLO → PIXEL
                x_center *= 500
                y_center *= 500
                bw *= 500
                bh *= 500

                xmin = x_center - bw / 2
                ymin = y_center - bh / 2

                xmax = x_center + bw / 2
                ymax = y_center + bh / 2

                # SHIFT
                xmin -= crop_x
                xmax -= crop_x

                ymin -= crop_y
                ymax -= crop_y

                # CLIP
                xmin = max(0, xmin)
                ymin = max(0, ymin)

                xmax = min(crop_size, xmax)
                ymax = min(crop_size, ymax)

                # INVALID BBOX
                if xmax <= xmin or ymax <= ymin:
                    continue

                # PIXEL → YOLO
                new_x = ((xmin + xmax) / 2) / crop_size
                new_y = ((ymin + ymax) / 2) / crop_size

                new_w = (xmax - xmin) / crop_size
                new_h = (ymax - ymin) / crop_size

                new_boxes.append([
                    new_x,
                    new_y,
                    new_w,
                    new_h
                ])

                new_labels.append(label)

            image = cropped
            bboxes = new_boxes
            class_labels = new_labels

        else:

            # Resize 224x224
            image = cv2.resize(image, (224, 224))

        # =====================================
        # 3. HORIZONTAL FLIP
        # =====================================

        if self.train and random.random() < 0.5:

            image = cv2.flip(image, 1)

            flipped_boxes = []

            for bbox in bboxes:

                x_center, y_center, bw, bh = bbox

                x_center = 1.0 - x_center

                flipped_boxes.append([
                    x_center,
                    y_center,
                    bw,
                    bh
                ])

            bboxes = flipped_boxes

        # =====================================
        # 4. COLOR JITTER
        # =====================================

        if self.train and random.random() < 0.5:

            alpha = random.uniform(0.5, 1.5)
            beta = random.randint(-30, 30)

            image = cv2.convertScaleAbs(
                image,
                alpha=alpha,
                beta=beta
            )

        # =====================================
        # 5. NORMALIZE
        # =====================================

        image = image.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        image = (image - mean) / std

        # =====================================
        # 6. TO TENSOR
        # =====================================

        image = torch.tensor(image).permute(2, 0, 1).float()

        return {
            'image': image,
            'bboxes': bboxes,
            'class_labels': class_labels
        }
    




# def safe_transform(self, image, bboxes, labels):

#     transformed = self.transform(
#         image=image,
#         bboxes=bboxes,
#         class_labels=labels
#     )

#     return transformed