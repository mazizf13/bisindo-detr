# =============================================
# COCO EVALUATION - STANDARD (NO THRESHOLD)
# =============================================

import contextlib
import io
import os
import json

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from model import DETR

# ===== CONFIGURATION =====
# MODEL_PATH    = "pretrained/mei/skenario1noaug.pt"
MODEL_PATH    = "pretrained/mei/skenario2.pt"
GT_JSON_PATH  = "data/mei/testing/coco.json"
IMAGE_ROOT    = "data/mei/testing/images"
PRED_JSON_PATH = "cocoresult/meiskenario2.json"

NUM_CLASSES   = 43    
NUM_QUERIES   = 25
INPUT_SIZE    = 224  # model input size

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ⚠️ NO CONFIDENCE_THRESHOLD — COCOeval handles ranking


# =============================================
# HELPER: rescale boxes from input_size to original image size
# =============================================
def rescale_boxes(pred_boxes: torch.Tensor, orig_w: int, orig_h: int, input_size: int = INPUT_SIZE):
    """
    pred_boxes : [N, 4]  format (xc, yc, w, h) normalized [0, 1]
    return     : [N, 4]  COCO format (x_min, y_min, w, h) in original pixels
    """
    # 1. Denormalize to input_size coordinates
    xc = pred_boxes[:, 0] * input_size
    yc = pred_boxes[:, 1] * input_size
    bw = pred_boxes[:, 2] * input_size
    bh = pred_boxes[:, 3] * input_size

    # 2. Convert center → corner (still in input_size space)
    x_min = xc - 0.5 * bw
    y_min = yc - 0.5 * bh
    x_max = xc + 0.5 * bw
    y_max = yc + 0.5 * bh

    # 3. Rescale to original image size
    scale_x = orig_w / input_size
    scale_y = orig_h / input_size

    x_min = x_min * scale_x
    y_min = y_min * scale_y
    x_max = x_max * scale_x
    y_max = y_max * scale_y

    # 4. COCO format: [x_min, y_min, width, height]
    coco_w = x_max - x_min
    coco_h = y_max - y_min

    return torch.stack([x_min, y_min, coco_w, coco_h], dim=1)


# =============================================
# MAIN
# =============================================
def main():
    print("=" * 60)
    print("COCO EVALUATION — PYCOCOTOOLS STANDARD")
    print("=" * 60)
    print(f"Model  : {MODEL_PATH}")
    print(f"GT JSON: {GT_JSON_PATH}")
    print(f"Device : {DEVICE}")
    print()

    # ----- LOAD MODEL -----
    model = DETR(num_classes=NUM_CLASSES, num_queries=NUM_QUERIES).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✅ Model loaded successfully.")

    # ----- LOAD GROUND TRUTH -----
    coco_gt  = COCO(GT_JSON_PATH)
    img_ids  = coco_gt.getImgIds()
    print(f"📸 Total test images: {len(img_ids)}")

    # Detect whether category_id starts from 0 or 1
    cats      = coco_gt.loadCats(coco_gt.getCatIds())
    min_cat   = min(cat["id"] for cat in cats)
    cat_offset = min_cat
    print(f"📋 JSON category IDs start from: {min_cat} (offset = {cat_offset})")

    # ----- PREPROCESSING TRANSFORM -----
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((INPUT_SIZE, INPUT_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])

    # ----- INFERENCE -----
    results = []

    print("\n⏳ Running inference...")
    with torch.no_grad():
        for idx, img_id in enumerate(img_ids):

            # Load image
            img_info  = coco_gt.loadImgs(img_id)[0]
            file_name = os.path.basename(img_info["file_name"])
            img_path  = os.path.join(IMAGE_ROOT, file_name)

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"  ⚠️  Failed to read: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img_rgb.shape[:2]

            # Preprocess
            img_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE) # type: ignore

            # Forward pass
            outputs = model(img_tensor)

            # Post-process
            pred_logits = outputs["pred_logits"][0]  # [Q, num_classes+1]
            pred_boxes  = outputs["pred_boxes"][0]   # [Q, 4]

            # Softmax → remove "no object" class
            probas = pred_logits.softmax(-1)[:, :-1]  # [Q, num_classes]

            # Best score and label per query
            scores, labels = probas.max(-1)

            # NO THRESHOLD — send ALL queries to COCOeval
            # COCOeval will handle ranking based on score

            # Rescale boxes to original image size
            boxes_coco = rescale_boxes(pred_boxes, orig_w, orig_h)

            # Collect results
            for box, score, label in zip(boxes_coco, scores, labels):
                results.append({
                    "image_id"   : int(img_id),
                    "category_id": int(label) + cat_offset,
                    "bbox"       : [round(float(v), 2) for v in box.tolist()],
                    "score"      : round(float(score), 6),
                })

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(img_ids)} images...")

    print(f"\n✅ Total predictions collected: {len(results)}")

    if len(results) == 0:
        print("❌ No predictions found. Check model / image path.")
        return

    # ----- SAVE PREDICTIONS -----
    os.makedirs(os.path.dirname(PRED_JSON_PATH), exist_ok=True)
    with open(PRED_JSON_PATH, "w") as f:
        json.dump(results, f)
    print(f"💾 Predictions saved to: {PRED_JSON_PATH}")

    # ----- COCO EVALUATION -----
    print("\n⏳ Running COCOeval...")

    coco_dt   = coco_gt.loadRes(PRED_JSON_PATH)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")

    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    s = coco_eval.stats

    # ----- PRINT RESULTS -----
    print("\n" + "=" * 60)
    print("PYCOCOTOOLS EVALUATION RESULTS SCENARIO 2")
    print("=" * 60)

    labels_map = [
        ("AP @[IoU=0.50:0.95] | area=all  | maxDets=100", s[0]),
        ("AP @[IoU=0.50     ] | area=all  | maxDets=100", s[1]),
        ("AP @[IoU=0.75     ] | area=all  | maxDets=100", s[2]),
        ("AP @[IoU=0.50:0.95] | area=small| maxDets=100", s[3]),
        ("AP @[IoU=0.50:0.95] | area=medium| maxDets=100", s[4]),
        ("AP @[IoU=0.50:0.95] | area=large| maxDets=100", s[5]),
        ("AR @[IoU=0.50:0.95] | area=all  | maxDets=1  ", s[6]),
        ("AR @[IoU=0.50:0.95] | area=all  | maxDets=10 ", s[7]),
        ("AR @[IoU=0.50:0.95] | area=all  | maxDets=100", s[8]),
        ("AR @[IoU=0.50:0.95] | area=small| maxDets=100", s[9]),
        ("AR @[IoU=0.50:0.95] | area=medium| maxDets=100", s[10]),
        ("AR @[IoU=0.50:0.95] | area=large| maxDets=100", s[11]),
    ]

    for name, val in labels_map:
        print(f"  {name} = {val:.4f}")

    print("=" * 60)

    print("\n📊 MAIN METRICS (thesis standard):")
    print(f"  AP @[IoU=0.50:0.95] : {s[0]:.4f}")
    print(f"  AP @[IoU=0.50]      : {s[1]:.4f}")
    print(f"  AP @[IoU=0.75]      : {s[2]:.4f}")
    print(f"  AR @[maxDets=1]     : {s[6]:.4f}")

    print("\n✅ EVALUATION FINISHED!")


if __name__ == "__main__":
    main()