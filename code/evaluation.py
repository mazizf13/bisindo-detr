# BELOM SESUAI SAMA PAPER DETR
# BELOM SESUAI SAMA PAPER DETR
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.data import DETRData
from src.model import DETR
from src.utils.boxes import box_cxcywh_to_xyxy
from src.utils.setup import get_classes

CLASSES = get_classes()
NUM_CLASSES = len(CLASSES)
MODEL_PATH = "pretrained/740_model.pt"   
DATA_PATH = "data/test"                  
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DETR(num_classes=NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

test_dataset = DETRData(DATA_PATH, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def box_iou(box1, box2):
    """Compute IoU between two boxes in xyxy format."""
    box1 = np.asarray(box1).flatten()
    box2 = np.asarray(box2).flatten()

    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, (float(box1[2]) - float(box1[0]))) * max(0, (float(box1[3]) - float(box1[1])))
    area2 = max(0, (float(box2[2]) - float(box2[0]))) * max(0, (float(box2[3]) - float(box2[1])))
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_ap(recall, precision):
    """Compute the average precision, given recall and precision curves."""
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = np.maximum(precision[i - 1], precision[i])
    indices = np.where(recall[1:] != recall[:-1])[0]
    ap = np.sum((recall[indices + 1] - recall[indices]) * precision[indices + 1])
    return ap


def evaluate(model, dataloader, classes, conf_threshold=0.5, iou_threshold=0.5):
    results = []
    class_metrics = {cls: {"tp": 0, "fp": 0, "fn": 0} for cls in range(len(classes))}
    all_preds, all_gts = [], []

    print("Running evaluation...\n")
    with torch.no_grad():
        for idx, (img_tensor, target) in enumerate(tqdm(dataloader)):
            img_tensor = img_tensor.to(device)

            # === Ground truth ===
            gt_labels = target["labels"].cpu().numpy()
            gt_boxes = target["boxes"].cpu().numpy()
            gt_boxes_xyxy = box_cxcywh_to_xyxy(torch.tensor(gt_boxes)).cpu().numpy()
            h, w = IMG_SIZE
            gt_boxes_xyxy_abs = gt_boxes_xyxy * np.array([w, h, w, h])
            gt_dicts = [
                {
                    "cat_id": int(label),
                    "bbox": np.asarray(bbox).flatten()
                }
                for label, bbox in zip(gt_labels, gt_boxes_xyxy_abs)
            ]

            all_gts.append(gt_dicts)

            # === Predictions ===
            outputs = model(img_tensor)
            pred_logits = outputs["pred_logits"].softmax(-1)[0, :, :-1]
            pred_boxes = outputs["pred_boxes"][0]

            pred_scores, pred_classes = torch.max(pred_logits, -1)
            keep = pred_scores > conf_threshold

            pred_scores = pred_scores[keep].cpu().numpy()
            pred_classes = pred_classes[keep].cpu().numpy()
            pred_boxes = pred_boxes[keep].cpu().numpy()
            pred_boxes_xyxy = box_cxcywh_to_xyxy(torch.tensor(pred_boxes)).cpu().numpy()
            pred_boxes_xyxy_abs = pred_boxes_xyxy * np.array([w, h, w, h])

            pred_dicts = [
                {"cat_id": int(c), "bbox": b, "score": float(s)}
                for c, b, s in zip(pred_classes, pred_boxes_xyxy_abs, pred_scores)
            ]
            all_preds.append(pred_dicts)

            # === Matching ===
            matched_gt_mask = np.zeros(len(gt_dicts), dtype=bool)
            correct = 0
            for pred in pred_dicts:
                pred_bbox = np.asarray(pred["bbox"]).flatten()
                best_iou, best_gt_idx = 0, -1
                for g_idx, gt in enumerate(gt_dicts):
                    gt_bbox = np.asarray(gt["bbox"]).flatten()
                    if matched_gt_mask[g_idx]:
                        continue
                    iou_val = box_iou(pred_bbox, gt_bbox)
                    if iou_val > best_iou:
                        best_iou, best_gt_idx = iou_val, g_idx
                if best_iou >= iou_threshold and best_gt_idx != -1 and pred["cat_id"] == gt_dicts[best_gt_idx]["cat_id"]:
                    class_metrics[pred["cat_id"]]["tp"] += 1
                    matched_gt_mask[best_gt_idx] = True
                    correct += 1
                else:
                    class_metrics[pred["cat_id"]]["fp"] += 1


            for g_idx, gt in enumerate(gt_dicts):
                if not matched_gt_mask[g_idx]:
                    class_metrics[gt["cat_id"]]["fn"] += 1

            results.append({"image_idx": idx, "total_gt": len(gt_dicts), "total_pred": len(pred_dicts), "correct": correct})

            print(f"[Image {idx}] GT={len(gt_dicts)}, Pred={len(pred_dicts)}, Correct={correct}")

    # === Per-class summary ===
    summary = {}
    aps = []
    for cls_id, cls_name in enumerate(classes):
        tp, fp, fn = class_metrics[cls_id]["tp"], class_metrics[cls_id]["fp"], class_metrics[cls_id]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # AP 
        recall_curve = np.array([0.0, recall, 1.0])
        precision_curve = np.array([0.0, precision, 0.0])
        ap = compute_ap(recall_curve, precision_curve)
        aps.append(ap)

        summary[cls_name] = {"precision": precision, "recall": recall, "f1": f1, "ap": ap, "tp": tp, "fp": fp, "fn": fn}

    mAP = np.mean(aps)

    print("\n=== Per-class Metrics ===")
    for cls, vals in summary.items():
        print(f"{cls}: P={vals['precision']:.4f}, R={vals['recall']:.4f}, "
              f"F1={vals['f1']:.4f}, AP={vals['ap']:.4f}, "
              f"TP={vals['tp']}, FP={vals['fp']}, FN={vals['fn']}")

    print(f"\n>>> mAP: {mAP:.4f}")

    # === Visualisasi TP/FP/FN ===
    tp_vals = [summary[c]["tp"] for c in classes]
    fp_vals = [summary[c]["fp"] for c in classes]
    fn_vals = [summary[c]["fn"] for c in classes]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].bar(classes, tp_vals, color="green")
    ax[0].set_title("True Positives per Class")

    ax[1].bar(classes, fp_vals, color="red")
    ax[1].set_title("False Positives per Class")

    ax[2].bar(classes, fn_vals, color="orange")
    ax[2].set_title("False Negatives per Class")

    plt.tight_layout()
    plt.show()

    # === Confusion Matrix ===
    all_pred_classes, all_gt_classes = [], []
    for preds, gts in zip(all_preds, all_gts):
        pred_classes = [p["cat_id"] for p in preds]
        gt_classes = [g["cat_id"] for g in gts]
        all_pred_classes.extend(pred_classes)
        all_gt_classes.extend(gt_classes)

    cm = confusion_matrix(all_gt_classes, all_pred_classes, labels=range(len(classes)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    return results, summary, mAP


if __name__ == "__main__":
    results, summary, mAP = evaluate(model, test_dataloader, CLASSES,
                                     conf_threshold=CONFIDENCE_THRESHOLD,
                                     iou_threshold=IOU_THRESHOLD)
