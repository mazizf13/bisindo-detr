import json
import numpy as np
import pandas as pd  # <--- Tambahan Wajib
from pycocotools.coco import COCO
from sklearn.metrics import classification_report

# === CONFIG ===
GT_JSON = "data/test_coco_result.json"
PRED_JSON = "data/temp_predict_eval.json"
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
OUTPUT_CSV = "laporan_klasifikasi_final.csv" # Nama file output
CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0]+box1[2], box2[0]+box2[2])
    y2 = min(box1[1]+box1[3], box2[1]+box2[3])
    inter_area = max(0, x2-x1) * max(0, y2-y1)
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def main():
    print("📊 Mengolah Laporan Klasifikasi...")
    coco_gt = COCO(GT_JSON)
    with open(PRED_JSON, 'r') as f:
        preds = json.load(f)

    # Mapping ID
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cats.sort(key=lambda x: x['name'])
    cat_id_to_index = {cat['id']: i for i, cat in enumerate(cats)}

    y_true = []
    y_pred = []
    
    img_ids = coco_gt.getImgIds()

    for img_id in img_ids:
        img_info = coco_gt.loadImgs(img_id)[0]
        W, H = img_info['width'], img_info['height']

        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(ann_ids)
        img_preds = [p for p in preds if p['image_id'] == img_id and p['score'] > CONF_THRESHOLD]
        
        used_preds = set()
        
        for gt in gt_anns:
            if gt['category_id'] not in cat_id_to_index: continue
            gt_idx = cat_id_to_index[gt['category_id']]x
            
            best_iou = 0
            best_pred_idx = -1
            
            for i, pred in enumerate(img_preds):
                if i in used_preds: continue
                
                p_box = list(pred['bbox'])
                if p_box[2] <= 1.0: 
                    p_box = [p_box[0]*W, p_box[1]*H, p_box[2]*W, p_box[3]*H]
                
                iou = compute_iou(gt['bbox'], p_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = i
            
            if best_iou >= IOU_THRESHOLD:
                pred_raw = img_preds[best_pred_idx]['category_id']
                y_true.append(gt_idx)
                y_pred.append(pred_raw)
                used_preds.add(best_pred_idx)
            else:
                # False Negative (Gagal Deteksi)
                y_true.append(gt_idx)
                y_pred.append(999) # Kode Gagal

    # --- CETAK & SIMPAN ---
    target_names = CLASSES + ["GAGAL_DETEKSI"]
    
    # Ubah 999 jadi index terakhir (26) biar masuk report
    y_pred_clean = [26 if x == 999 else x for x in y_pred]
    
    # 1. Print Text Biasa
    print("\n=== PREVIEW LAPORAN ===")
    print(classification_report(y_true, y_pred_clean, target_names=target_names, zero_division=0))
    
    # 2. Simpan ke CSV (Ini intinya)
    # output_dict=True bikin hasilnya jadi Dictionary, bukan Teks
    report_dict = classification_report(y_true, y_pred_clean, target_names=target_names, zero_division=0, output_dict=True)
    
    # Convert ke Pandas DataFrame
    df = pd.DataFrame(report_dict).transpose()
    
    # Simpan
    df.to_csv(OUTPUT_CSV)
    print(f"\n✅ Laporan tersimpan di file: {OUTPUT_CSV}")
    print("   (Bisa langsung dibuka di Excel)")

if __name__ == "__main__":
    main()