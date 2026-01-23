import torch
from torch.utils.data import DataLoader
import pandas as pd
from data import DETRData, stacker
from model import DETR
from loss import HungarianMatcher
from utils.setup import get_classes

# =========================
# CONFIG
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = "data/train"  # sesuaikan path dataset
BATCH_SIZE = 4
NUM_CLASSES = 26

# =========================
# LOAD DATA
# =========================
train_dataset = DETRData(DATA_PATH, train=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=stacker)

# =========================
# INISIALISASI MODEL
# =========================
model = DETR(num_classes=NUM_CLASSES).to(DEVICE)
model.eval()  # evaluasi

# =========================
# HUNGARIAN MATCHER
# =========================
weight_dict = {'class_weighting': 1, 'bbox_weighting': 5, 'giou_weighting': 2}
matcher = HungarianMatcher(weight_dict)

# =========================
# EVALUATION FUNCTION
# =========================
def box_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x1_min = x1 - w1 / 2
    y1_min = y1 - h1 / 2
    x1_max = x1 + w1 / 2
    y1_max = y1 + h1 / 2
    x2_min = x2 - w2 / 2
    y2_min = y2 - h2 / 2
    x2_max = x2 + w2 / 2
    y2_max = y2 + h2 / 2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def evaluate_model(model, dataloader, device, matcher, num_classes=26):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_iou = 0.0
    num_boxes = 0

    per_class_correct = {i: 0 for i in range(num_classes)}
    per_class_total = {i: 0 for i in range(num_classes)}
    per_class_iou_sum = {i: 0.0 for i in range(num_classes)}
    per_class_iou_count = {i: 0 for i in range(num_classes)}

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = [{k: v.to(device) for k, v in target.items()} for target in y]
            yhat = model(X)

            for i in range(len(y)):
                pred_logits = yhat["pred_logits"][i]
                pred_boxes = yhat["pred_boxes"][i]
                tgt_labels = y[i]["labels"]
                tgt_boxes = y[i]["boxes"]

                yhat_sample = {"pred_logits": pred_logits.unsqueeze(0), "pred_boxes": pred_boxes.unsqueeze(0)}
                y_sample = [{"labels": tgt_labels, "boxes": tgt_boxes}]
                indices = matcher(yhat_sample, y_sample)
                pred_indices, tgt_indices = indices[0]

                for p, t in zip(pred_indices, tgt_indices):
                    pred_class = pred_logits[p].argmax().item()
                    tgt_class = tgt_labels[t].item()

                    per_class_total[tgt_class] += 1
                    if pred_class == tgt_class:
                        total_correct += 1
                        per_class_correct[pred_class] += 1

                    iou = box_iou(pred_boxes[p].cpu(), tgt_boxes[t].cpu())
                    total_iou += iou
                    num_boxes += 1
                    per_class_iou_sum[tgt_class] += iou
                    per_class_iou_count[tgt_class] += 1

                total_samples += len(tgt_labels)

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_iou = total_iou / num_boxes if num_boxes > 0 else 0

    per_class_metrics = []
    for cls in range(num_classes):
        acc = per_class_correct[cls] / per_class_total[cls] if per_class_total[cls] > 0 else 0
        avg_iou_cls = per_class_iou_sum[cls] / per_class_iou_count[cls] if per_class_iou_count[cls] > 0 else 0
        per_class_metrics.append({
            "Class": cls,
            "Accuracy": acc,
            "Avg IoU": avg_iou_cls,
            "Samples": per_class_total[cls],
        })

    return accuracy, avg_iou, per_class_metrics

# =========================
# RUN EVALUATION
# =========================
accuracy, avg_iou, per_class_metrics = evaluate_model(model, train_loader, DEVICE, matcher, NUM_CLASSES)
print(f"\nTrain Results:\nClassification Accuracy: {accuracy:.4f}\nAverage IoU: {avg_iou:.4f}\n")
df = pd.DataFrame(per_class_metrics)
print(df.to_string(index=False))
