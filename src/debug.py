import json
import numpy as np
from pycocotools.coco import COCO

# === CONFIG ===
GT_JSON = "data/test_coco_result.json"
PRED_JSON = "data/temp_predictions.json" # Pastikan path ini benar! (kadang ada di folder data/)
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
    
    if union_area == 0: return 0
    return inter_area / union_area

def main():
    print("🕵️‍♂️ MEMULAI DIAGNOSA...")
    coco_gt = COCO(GT_JSON)
    with open(PRED_JSON, 'r') as f:
        preds = json.load(f)
    
    print(f"📄 Jumlah Prediksi: {len(preds)}")
    
    # Cek 5 Gambar Pertama Saja
    img_ids = coco_gt.getImgIds()[:5] 
    
    total_iou_sum = 0
    count = 0

    for img_id in img_ids:
        print(f"\n📸 Cek Image ID: {img_id}")
        
        # Ambil GT
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(ann_ids)
        
        # Ambil Prediksi
        img_preds = [p for p in preds if p['image_id'] == img_id]
        print(f"   -> Punya {len(gt_anns)} GT dan {len(img_preds)} Prediksi")

        for gt in gt_anns:
            gt_box = gt['bbox']
            gt_cls = gt['category_id']
            print(f"   🎯 GT Box: {gt_box} (Kelas: {gt_cls})")
            
            best_iou = 0
            for pred in img_preds:
                p_box = pred['bbox']
                iou = compute_iou(gt_box, p_box)
                
                # Simpan IoU terbaik
                if iou > best_iou:
                    best_iou = iou
                    
                # Debug print kalo IoU > 0.1 (biar gak nyampah)
                if iou > 0.1:
                    print(f"      ❓ Kandidat IoU: {iou:.4f} (Score: {pred['score']:.2f})")
            
            print(f"      👉 BEST IoU untuk GT ini: {best_iou:.4f}")
            
            if best_iou > 0.0:
                total_iou_sum += best_iou
                count += 1

    if count > 0:
        print(f"\n📊 Rata-rata Best IoU (dari 5 gambar): {total_iou_sum/count:.4f}")
    else:
        print("\n❌ TIDAK ADA IoU > 0. Koordinat mungkin salah total!")

if __name__ == "__main__":
    main()