import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

# Paths
GT_JSON_PATH = "data/asli/test/result.json"
PRED_JSON_PATH = "evaluation_result/110226_noaug300.json"

# Load COCO GT & Prediction
coco_gt = COCO(GT_JSON_PATH)
coco_dt = coco_gt.loadRes(PRED_JSON_PATH)

# COCO Evaluation
coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()  # ini tetap bisa tampilkan mAP global

# ===============================================
# Ambil AP, Precision, Recall per Kelas
# ===============================================

num_classes = len(coco_eval.params.catIds)
class_ids = coco_eval.params.catIds  # daftar category_id
class_names = [coco_gt.cats[i]['name'] for i in class_ids]

ap_per_class = []
precision_per_class = []
recall_per_class = []

for k in range(num_classes):
    # precision tensor shape: [IoU, Recall, class, area, maxDet]
    p = coco_eval.eval['precision'][:, :, k, 0, 2]  # area=0(all), maxDet=2(100)
    p = p[p > -1]  # buang nilai -1 (kelas tanpa deteksi)
    ap = np.mean(p) if len(p) > 0 else float('nan')
    ap_per_class.append(ap)

    # Rata-rata Precision & Recall
    # Recall tensor shape: [IoU, class, area, maxDet]
    r = coco_eval.eval['recall'][:, k, 0, 2]
    r = r[r > -1]
    precision_per_class.append(ap)  # di COCO, AP = rata-rata precision di semua IoU & recall
    recall_per_class.append(np.mean(r) if len(r) > 0 else float('nan'))

# ===============================================
# Print hasil
# ===============================================
print("Hasil Evaluasi per Kelas:")
print("{:<10} {:<8} {:<10} {:<10}".format("Kelas", "ID", "AP", "Recall"))
for i, name in enumerate(class_names):
    print("{:<10} {:<8} {:<10.3f} {:<10.3f}".format(name, class_ids[i], ap_per_class[i], recall_per_class[i]))

# opsional: simpan ke JSON
results_dict = {name: {"category_id": class_ids[i],
                       "AP": ap_per_class[i],
                       "Recall": recall_per_class[i]} 
                for i, name in enumerate(class_names)}

with open("evaluation_result/per_class_metrics.json", "w") as f:
    json.dump(results_dict, f, indent=4)