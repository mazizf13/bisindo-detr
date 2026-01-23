import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools.coco import COCO
from sklearn.metrics import confusion_matrix

# === CONFIG ===
GT_JSON = "data/test_coco_result.json"
PRED_JSON = "data/temp_predict_eval.json"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

def compute_iou(box1, box2):
    # box format: [x_min, y_min, w, h]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0]+box1[2], box2[0]+box2[2])
    y2 = min(box1[1]+box1[3], box2[1]+box2[3])
    
    inter_w = max(0, x2-x1)
    inter_h = max(0, y2-y1)
    inter_area = inter_w * inter_h
    
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    
    union_area = box1_area + box2_area - inter_area
    if union_area == 0: return 0
    return inter_area / union_area

def main():
    coco_gt = COCO(GT_JSON)
    with open(PRED_JSON, 'r') as f:
        preds = json.load(f)
        
    # Ensure ID mapping is correct (A=0, B=1, ...)
    # We assume your model outputs 0-25.
    # We need to check what IDs are in the GT json.
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cats.sort(key=lambda x: x['name']) # Sort by name A-Z
    
    # Map GT IDs to 0-25 index
    gt_id_to_index = {cat['id']: i for i, cat in enumerate(cats)}
    
    y_true = []
    y_pred = []
    
    img_ids = coco_gt.getImgIds()
    
    print(f"Processing {len(img_ids)} images...")
    
    for img_id in img_ids:
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(ann_ids)
        
        # Get predictions for this image
        img_preds = [p for p in preds if p['image_id'] == img_id and p['score'] > CONF_THRESHOLD]
        
        used_preds = set()
        
        for gt in gt_anns:
            if gt['category_id'] not in gt_id_to_index: continue
            gt_idx = gt_id_to_index[gt['category_id']]
            
            best_iou = 0
            best_pred_idx = -1
            
            for i, pred in enumerate(img_preds):
                if i in used_preds: continue
                
                iou = compute_iou(gt['bbox'], pred['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = i
            
            if best_iou >= IOU_THRESHOLD:
                # Match found!
                pred_cls = img_preds[best_pred_idx]['category_id']
                y_true.append(gt_idx)
                y_pred.append(pred_cls)
                used_preds.add(best_pred_idx)
    
    # Plotting
    if not y_true:
        print("No matches found! Check thresholds or coordinates.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title(f'Confusion Matrix (Matches: {len(y_true)})')
    plt.tight_layout()
    plt.savefig('confusion_matrix_fixed.png')
    plt.show()

if __name__ == '__main__':
    main()

# PERLU DICEK LAGI APAKAH SUDAH SESUAI DENGAN YANG DIINGINKAN
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pycocotools.coco import COCO
# from sklearn.metrics import confusion_matrix

# # === CONFIG ===
# GT_JSON = "data/test_coco_result.json"
# PRED_JSON = "data/temp_predict_eval.json"
# CONF_THRESHOLD = 0.5
# IOU_THRESHOLD = 0.5
# CLASSES = [
#     "A","B","C","D","E","F","G","H","I","J","K","L","M",
#     "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
# ]
# EXTRA_LABEL = "No Prediction"  # opsional untuk unmatched GT / pred

# def compute_iou(box1, box2):
#     # box format: [x_min, y_min, w, h]
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[0]+box1[2], box2[0]+box2[2])
#     y2 = min(box1[1]+box1[3], box2[1]+box2[3])

#     inter_w = max(0, x2-x1)
#     inter_h = max(0, y2-y1)
#     inter_area = inter_w * inter_h

#     box1_area = box1[2] * box1[3]
#     box2_area = box2[2] * box2[3]

#     union_area = box1_area + box2_area - inter_area
#     if union_area == 0: return 0
#     return inter_area / union_area

# def main():
#     # Load GT & pred JSON
#     coco_gt = COCO(GT_JSON)
#     with open(PRED_JSON, 'r') as f:
#         preds = json.load(f)

#     # Map GT category_id → 0..25 index
#     cats = coco_gt.loadCats(coco_gt.getCatIds())
#     cats.sort(key=lambda x: x['name'])  # Sort alphabetically
#     gt_id_to_index = {cat['id']: i for i, cat in enumerate(cats)}

#     y_true = []
#     y_pred = []

#     img_ids = coco_gt.getImgIds()
#     print(f"Processing {len(img_ids)} images...")

#     for img_id in img_ids:
#         ann_ids = coco_gt.getAnnIds(imgIds=img_id)
#         gt_anns = coco_gt.loadAnns(ann_ids)

#         # Prediksi untuk image ini, filter confidence
#         img_preds = [p for p in preds if p['image_id']==img_id and p['score']>CONF_THRESHOLD]

#         used_preds = set()

#         # Match GT → pred
#         for gt in gt_anns:
#             if gt['category_id'] not in gt_id_to_index: continue
#             gt_idx = gt_id_to_index[gt['category_id']]

#             best_iou = 0
#             best_pred_idx = -1
#             for i, pred in enumerate(img_preds):
#                 if i in used_preds: continue
#                 iou = compute_iou(gt['bbox'], pred['bbox'])
#                 if iou > best_iou:
#                     best_iou = iou
#                     best_pred_idx = i

#             if best_iou >= IOU_THRESHOLD:
#                 pred_cls_id = img_preds[best_pred_idx]['category_id']
#                 pred_idx = gt_id_to_index.get(pred_cls_id, -1)
#                 if pred_idx != -1:
#                     y_true.append(gt_idx)
#                     y_pred.append(pred_idx)
#                 used_preds.add(best_pred_idx)
#             else:
#                 # unmatched GT → bisa dianggap No Prediction
#                 y_true.append(gt_idx)
#                 y_pred.append(len(CLASSES))  # extra column

#         # unmatched preds → bisa dianggap false positive
#         for i, pred in enumerate(img_preds):
#             if i in used_preds:
#                 continue
#             pred_cls_id = pred['category_id']
#             pred_idx = gt_id_to_index.get(pred_cls_id, -1)
#             if pred_idx != -1:
#                 y_true.append(len(CLASSES))  # extra row
#                 y_pred.append(pred_idx)

#     # Labels untuk confusion matrix
#     labels = CLASSES + [EXTRA_LABEL]

#     cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))

#     plt.figure(figsize=(22, 20))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=labels, yticklabels=labels)
#     plt.xlabel('Predicted')
#     plt.ylabel('Ground Truth')
#     plt.title(f'Confusion Matrix (Matches: {len(y_true)})')
#     plt.tight_layout()
#     plt.savefig('confusion_matrix_fixeddd.png')
#     plt.show()
#     print("✅ Confusion matrix saved as confusion_matrix_fixeddd.png")

# if __name__ == "__main__":
#     main()
