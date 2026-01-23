import json
import os
from pycocotools.coco import COCO

# === CONFIG ===
GT_JSON = "data/test_coco_result.json"
PRED_JSON = "data/temp_predict_eval.json"
TARGET_CLASS = "M" # Huruf yang mau dicari
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

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
    coco = COCO(GT_JSON)
    with open(PRED_JSON, 'r') as f:
        preds = json.load(f)

    # Cari ID untuk kelas M
    cats = coco.loadCats(coco.getCatIds())
    cats.sort(key=lambda x: x['name'])
    target_cat_id = next(cat['id'] for cat in cats if cat['name'] == TARGET_CLASS)
    
    print(f"🕵️‍♂️ Mencari Gambar '{TARGET_CLASS}' yang hilang...")
    
    # Ambil semua gambar yang aslinya (GT) adalah M
    target_img_ids = []
    for img_id in coco.getImgIds():
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            if ann['category_id'] == target_cat_id:
                target_img_ids.append(img_id)
                break
    
    print(f"   Total Gambar GT '{TARGET_CLASS}': {len(target_img_ids)}")

    # Cek satu-satu
    for img_id in target_img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_name = os.path.basename(img_info['file_name'])
        W, H = img_info['width'], img_info['height']
        
        # Ambil GT box
        ann_ids = coco.getAnnIds(imgIds=img_id)
        gt_box = coco.loadAnns(ann_ids)[0]['bbox']
        
        # Ambil Prediksi
        curr_preds = [p for p in preds if p['image_id'] == img_id]
        
        # Analisis Kegagalan
        status = "❌ HILANG"
        reason = "Tidak ada prediksi sama sekali"
        
        if len(curr_preds) > 0:
            best_score = 0
            best_iou = 0
            
            for p in curr_preds:
                # Fix koordinat jika perlu
                p_box = list(p['bbox'])
                if p_box[2] <= 1.0: 
                    p_box = [p_box[0]*W, p_box[1]*H, p_box[2]*W, p_box[3]*H]
                
                iou = compute_iou(gt_box, p_box)
                score = p['score']
                
                if score > best_score: best_score = score
                if iou > best_iou: best_iou = iou
            
            if best_score < CONF_THRESHOLD:
                reason = f"Score terlalu rendah ({best_score:.2f} < {CONF_THRESHOLD})"
            elif best_iou < IOU_THRESHOLD:
                reason = f"Posisi melenceng (IoU {best_iou:.2f} < {IOU_THRESHOLD})"
            else:
                status = "✅ AMAN"
                reason = "Terdeteksi"

        if status == "❌ HILANG":
            print(f"\n⚠️ MASALAH DITEMUKAN DI FILE: {file_name}")
            print(f"   Alasan: {reason}")

if __name__ == "__main__":
    main()