import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycocotools.coco import COCO
from sklearn.metrics import confusion_matrix

# === CONFIG ===
GT_JSON = "data/test_coco_result.json"       # Kunci Jawaban (Ground Truth)
PRED_JSON = "data/tempp_predictions.json"     # Hasil Prediksi Model
CONF_THRESHOLD = 0.25                        # Threshold confidence score (bisa disesuaikan)
IOU_THRESHOLD = 0.1                         # Threshold IoU untuk dianggap "match"
CLASSES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
]
# =================

def compute_iou(box1, box2):
    """
    Hitung IoU dua kotak.
    Format kotak diharapkan: [x_min, y_min, width, height] (Standard COCO)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])
    
    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height
    
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0
        
    return inter_area / union_area

def main():
    print("📊 Memulai Analisis Confusion Matrix...")
    
    # 1. Load Ground Truth
    try:
        coco_gt = COCO(GT_JSON)
    except Exception as e:
        print(f"❌ Gagal memuat Ground Truth JSON: {e}")
        return

    # 2. Load Prediksi
    try:
        with open(PRED_JSON, 'r') as f:
            preds = json.load(f)
        print(f"✅ Memuat {len(preds)} prediksi dari {PRED_JSON}")
    except FileNotFoundError:
        print(f"❌ Error: File {PRED_JSON} tidak ditemukan! Pastikan path benar.")
        return

    # 3. Mapping Category ID (PENTING!)
    # Kita urutkan kategori berdasarkan nama ('A', 'B', ...) agar index 0 = A, 1 = B, dst.
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cats.sort(key=lambda x: x['name'])
    
    # Mapping dari ID di JSON ke Index 0-25
    cat_id_to_index = {cat['id']: i for i, cat in enumerate(cats)}
    
    # Mapping dari Index 0-25 ke Nama Kelas (untuk display/debug)
    index_to_name = {i: cat['name'] for i, cat in enumerate(cats)}

    print("\n🔍 Cek Mapping ID Kelas:")
    print(f"   Total Kelas di JSON GT: {len(cats)}")
    if len(cats) > 0:
        print(f"   Contoh: '{cats[0]['name']}' (ID JSON: {cats[0]['id']}) --> Index Matrix: {cat_id_to_index[cats[0]['id']]}")

    y_true = []
    y_pred = []
    
    img_ids = coco_gt.getImgIds()
    match_count = 0
    
    print(f"\n⏳ Sedang mencocokkan kotak di {len(img_ids)} gambar...")

    for img_id in img_ids:
        # Ambil Info Gambar (Penting buat Un-normalize koordinat prediksi)
        img_info = coco_gt.loadImgs(img_id)[0]
        W_img = img_info['width']
        H_img = img_info['height']

        # Ambil Ground Truth (Annotations) untuk gambar ini
        ann_ids = coco_gt.getAnnIds(imgIds=img_id)
        gt_anns = coco_gt.loadAnns(ann_ids)
        
        # Ambil Prediksi untuk gambar ini & Filter Score
        # Asumsi: 'image_id' di preds cocok dengan 'id' di img_ids COCO
        img_preds = [p for p in preds if p['image_id'] == img_id and p['score'] > CONF_THRESHOLD]
        
        used_preds = set()
        
        for gt in gt_anns:
            gt_id_json = gt['category_id']
            
            # Skip jika kategori GT tidak ada di mapping kita (misal background/unknown)
            if gt_id_json not in cat_id_to_index:
                continue
                
            gt_idx = cat_id_to_index[gt_id_json]
            
            best_iou = 0
            best_pred_idx = -1
            
            # Cari prediksi terbaik untuk GT ini
            for i, pred in enumerate(img_preds):
                if i in used_preds: 
                    continue
                
                # --- FIX KOORDINAT DI SINI ---
                p_box = list(pred['bbox']) # Copy biar aman
                
                # Deteksi & Fix Normalisasi
                # Jika width & height <= 1.0, kemungkinan besar ini normalized (0-1).
                # Kita kalikan dengan dimensi gambar asli.
                # Note: Kadang x,y bisa > 1 tapi w,h < 1 (jarang di COCO format, tapi worth checking w/h)
                if p_box[2] <= 1.0 and p_box[3] <= 1.0:
                    p_box[0] *= W_img
                    p_box[1] *= H_img
                    p_box[2] *= W_img
                    p_box[3] *= H_img
                
                # Hitung IoU
                iou = compute_iou(gt['bbox'], p_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = i
            
            # Penentuan Match
            if best_iou >= IOU_THRESHOLD:
                # Ambil category_id dari prediksi
                pred_id_raw = img_preds[best_pred_idx]['category_id']
                
                # ASUMSI PENTING:
                # Model Anda outputnya index kelas 0-25 (langsung urutan kelas).
                # Jika model output ID JSON, perlu di-map balik.
                # Biasanya model DETR outputnya raw index (0, 1, 2...).
                # Jadi pred_idx = pred_id_raw.
                
                pred_idx = pred_id_raw
                
                # Safety check: pastikan index prediksi valid (0-25)
                if 0 <= pred_idx < len(CLASSES):
                    y_true.append(gt_idx)
                    y_pred.append(pred_idx)
                    match_count += 1
                    used_preds.add(best_pred_idx)
                else:
                    print(f"⚠️ Warning: Prediksi index {pred_idx} di luar range kelas!")

    print(f"✅ Selesai! Menemukan {match_count} pasangan (Matches) dari total Ground Truth.")

    if len(y_true) == 0:
        print("❌ Masih 0 Matches. Kemungkinan penyebab:")
        print("   1. Koordinat prediksi masih salah skala.")
        print("   2. ID Image di prediksi tidak cocok dengan ID Image di GT.")
        print("   3. Threshold IoU/Score terlalu tinggi.")
        return

    # --- PLOTTING ---
    # Pastikan labels mencakup semua kelas agar matriks ukuran tetap 26x26
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))

    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASSES, yticklabels=CLASSES)
    
    plt.xlabel('Prediksi Model', fontsize=14)
    plt.ylabel('Kunci Jawaban (Ground Truth)', fontsize=14)
    plt.title(f'Confusion Matrix (IoU > {IOU_THRESHOLD}, Score > {CONF_THRESHOLD})\nTotal Matches: {match_count}', fontsize=16)
    plt.tight_layout()
    plt.savefig('confusion_matrix_final.png')
    print("✅ Gambar tersimpan sebagai 'confusion_matrix_final.png'")
    plt.show()

if __name__ == "__main__":
    main()