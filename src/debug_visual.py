import torch
import cv2
import os
import sys
import json
import random
import matplotlib.pyplot as plt
import torchvision.transforms as T
from pycocotools.coco import COCO

# Tambahkan path src agar import aman
sys.path.append("src")

try:
    from src.model import DETR
except ImportError:
    from model import DETR

# === CONFIG ===
MODEL_PATH = "pretrained/999_model.pt"
GT_JSON_PATH = "data/temp_predict_eval.json"
IMAGE_ROOT = "data/test_ood/images"
NUM_CLASSES = 26
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Warna (BGR format untuk OpenCV)
COLOR_GT = (0, 255, 0)    # Hijau
COLOR_PRED = (0, 0, 255)  # Merah

def draw_text_with_bg(img, text, x, y, color, bg_color=(0,0,0), font_scale=0.6):
    """Fungsi bantu biar teks ada background hitamnya (biar kebaca)"""
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    
    # Gambar kotak background
    # Pastikan koordinat tidak negatif
    y_bg_top = max(0, y - text_height - 5)
    x_bg_left = max(0, x)
    
    cv2.rectangle(img, (x_bg_left, y_bg_top), (x + text_width, y + 5), bg_color, -1)
    # Tulis teks
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def main():
    # 1. Load Model
    print("🚀 Loading Model...")
    model = DETR(num_classes=NUM_CLASSES, num_queries=25).to(DEVICE) 
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print("✅ Model siap.")

    # 2. Setup Data COCO
    coco = COCO(GT_JSON_PATH)
    all_img_ids = coco.getImgIds()
    
    # --- AMBIL 4 GAMBAR ACAK ---
    num_samples = min(4, len(all_img_ids))
    selected_ids = random.sample(all_img_ids, num_samples)
    print(f"🎲 Mengambil {num_samples} gambar acak ID: {selected_ids}")

    # 3. Setup Preprocessing
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)), 
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])

    # 4. Setup Plot (2x2 Grid)
    # figsize=(15, 15) cukup besar agar gambar jelas
    fig, axs = plt.subplots(2, 2, figsize=(15, 15))
    axs = axs.flatten()

    # --- LOOPING 4 GAMBAR ---
    for i, img_id in enumerate(selected_ids):
        ax = axs[i]
        
        # A. Load Info & Gambar
        img_info = coco.loadImgs(img_id)[0]
        file_name = os.path.basename(img_info['file_name']) 
        img_path = os.path.join(IMAGE_ROOT, file_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Gagal baca: {file_name}")
            ax.text(0.5, 0.5, "Image Not Found", ha='center')
            ax.axis('off')
            continue
            
        h_orig, w_orig, _ = img.shape
        img_vis = img.copy() 

        # B. GAMBAR GROUND TRUTH (HIJAU)
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        for ann in anns:
            box = ann['bbox']
            cat_id = ann['category_id']
            x, y, w, h = map(int, box)
            
            # Kotak Hijau
            cv2.rectangle(img_vis, (x, y), (x+w, y+h), COLOR_GT, 3)
            # Label di ATAS kotak
            draw_text_with_bg(img_vis, f"GT:{cat_id}", x, y - 10, COLOR_GT)

        # C. PREDIKSI MODEL (MERAH)
        img_rgb_input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = preprocess(img_rgb_input).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(img_tensor)

        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > 0.5 
        
        bboxes_scaled = outputs['pred_boxes'][0, keep]
        labels = probas[keep].max(-1).indices
        scores = probas[keep].max(-1).values

        if len(bboxes_scaled) == 0:
            draw_text_with_bg(img_vis, "NO DETECTION", 50, 50, (0,0,255), (255,255,255), 1.5)

        for box, label, score in zip(bboxes_scaled, labels, scores):
            cx, cy, w, h = box.tolist()
            cx, cy, w, h = cx * w_orig, cy * h_orig, w * w_orig, h * h_orig
            x = int(cx - 0.5 * w)
            y = int(cy - 0.5 * h)
            w_px = int(w)
            h_px = int(h)
            
            cv2.rectangle(img_vis, (x, y), (x+w_px, y+h_px), COLOR_PRED, 3)
            label_text = f"Pred:{label.item()} ({score:.2f})"
            
            # Label di BAWAH kotak, dengan sedikit jarak
            draw_text_with_bg(img_vis, label_text, x, y + h_px + 25, COLOR_PRED, (255,255,255))

        # D. Tampilkan di Subplot
        img_final = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
        ax.imshow(img_final)
        
        # Set Title dengan Padding (Jarak Judul ke Gambar)
        ax.set_title(f"File: {file_name}", fontsize=12, pad=15) 
        ax.axis('off')

    # --- PENGATURAN JARAK (INI KUNCINYA) ---
    # hspace=0.3 memberika jarak vertikal 30% dari tinggi plot
    # wspace=0.1 memberikan jarak horizontal
    plt.subplots_adjust(wspace=0.1, hspace=0.3) 
    
    plt.show()

if __name__ == "__main__":
    main()


# Single Image Debug Visualization
# import torch
# import cv2
# import os
# import sys
# import json
# import matplotlib.pyplot as plt
# import torchvision.transforms as T  # <--- Tambahan Wajib
# from pycocotools.coco import COCO

# # Tambahkan path src agar import aman
# sys.path.append("src")

# try:
#     from src.model import DETR
# except ImportError:
#     from model import DETR

# # === CONFIG ===
# MODEL_PATH = "pretrained/999_model.pt"
# GT_JSON_PATH = "data/test_coco_result.json"
# IMAGE_ROOT = "data/test/images"
# NUM_CLASSES = 26
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Warna: Hijau = Kunci Jawaban (GT), Merah = Prediksi Model
# COLOR_GT = (0, 255, 0) 
# COLOR_PRED = (255, 0, 0)

# def main():
#     # 1. Load Model
#     print("Load Model...")
#     # INGAT: Pakai num_queries=25 sesuai error log tadi
#     model = DETR(num_classes=NUM_CLASSES, num_queries=25).to(DEVICE) 
    
#     checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
#     if 'model_state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['model_state_dict'])
#     elif 'model' in checkpoint:
#         model.load_state_dict(checkpoint['model'])
#     else:
#         model.load_state_dict(checkpoint)
        
#     model.eval()
#     print("✅ Model siap.")

#     # 2. Ambil 1 Gambar dari JSON buat sampel
#     coco = COCO(GT_JSON_PATH)
#     img_ids = coco.getImgIds()
    
#     # Ganti index [0] ke [1], [2], dst jika ingin cek gambar lain
#     img_id = img_ids[0] 
#     img_info = coco.loadImgs(img_id)[0]
    
#     # Fix path dari Label Studio
#     file_name = os.path.basename(img_info['file_name']) 
#     img_path = os.path.join(IMAGE_ROOT, file_name)

#     print(f"Mengecek Gambar: {file_name}")

#     # 3. Buka Gambar
#     img = cv2.imread(img_path)
#     if img is None:
#         print(f"❌ Error: Gagal membaca gambar di {img_path}")
#         return
        
#     # Convert ke RGB untuk Matplotlib & Model
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     h_orig, w_orig, _ = img.shape

#     # 4. GAMBAR KOTAK KUNCI JAWABAN (GT) - HIJAU
#     ann_ids = coco.getAnnIds(imgIds=img_id)
#     anns = coco.loadAnns(ann_ids)
    
#     print("\n--- DATA GROUND TRUTH (JSON) ---")
#     # Kita copy gambar dulu biar bersih
#     img_vis = img_rgb.copy()
    
#     for ann in anns:
#         box = ann['bbox'] # [x, y, w, h]
#         cat_id = ann['category_id']
#         print(f"GT Box: {box}, Kelas ID: {cat_id}")
        
#         # Gambar kotak Hijau
#         x, y, w, h = map(int, box)
#         cv2.rectangle(img_vis, (x, y), (x+w, y+h), COLOR_GT, 2)
#         cv2.putText(img_vis, f"GT:{cat_id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GT, 2)

#     # 5. PREPROCESSING (BAGIAN YANG DIBENARKAN)
#     # Harus SAMA PERSIS dengan data.py saat training
#     preprocess = T.Compose([
#         T.ToPILImage(),
#         T.Resize((224, 224)), # Resize ke ukuran training
#         T.ToTensor(),
#         # Normalisasi ImageNet (PENTING BANGET)
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
#     ])

#     # Masukkan gambar ke transform
#     img_tensor = preprocess(img_rgb).unsqueeze(0).to(DEVICE)

#     # 6. PREDIKSI MODEL
#     with torch.no_grad():
#         outputs = model(img_tensor)

#     probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    
#     # Threshold Visualisasi (Bisa dinaikkan kalau terlalu rame)
#     keep = probas.max(-1).values > 0.5 
    
#     bboxes_scaled = outputs['pred_boxes'][0, keep]
#     labels = probas[keep].max(-1).indices
#     scores = probas[keep].max(-1).values

#     print("\n--- DATA PREDIKSI MODEL ---")
#     if len(bboxes_scaled) == 0:
#         print("MODEL TIDAK MENDETEKSI APAPUN!")
    
#     for box, label, score in zip(bboxes_scaled, labels, scores):
#         # Un-normalize Box
#         # Output DETR adalah relative (0-1).
#         # Kita kalikan dengan UKURAN ASLI gambar (w_orig, h_orig)
#         # Logikanya: 50% di gambar 224x224 sama posisinya dengan 50% di gambar asli.
        
#         cx, cy, w, h = box.tolist()
#         cx, cy, w, h = cx * w_orig, cy * h_orig, w * w_orig, h * h_orig
        
#         x = int(cx - 0.5 * w)
#         y = int(cy - 0.5 * h)
#         w = int(w)
#         h = int(h)
        
#         print(f"Pred Box: {[x, y, w, h]}, Kelas ID: {label.item()}, Score: {score.item():.2f}")

#         # Gambar kotak Merah
#         cv2.rectangle(img_vis, (x, y), (x+w, y+h), COLOR_PRED, 2)
#         cv2.putText(img_vis, f"Pred:{label.item()} ({score:.2f})", (x, y+15), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRED, 2)

#     # 7. Tampilkan
#     plt.figure(figsize=(12, 12))
#     plt.imshow(img_vis)
#     plt.title(f"Gambar: {file_name}\nHijau = Kunci Jawaban | Merah = Prediksi Model")
#     plt.axis('off')
#     plt.show()

# if __name__ == "__main__":
#     main()

 
# 4 citra test random
# import torch
# import cv2
# import os
# import sys
# import json
# import random
# import matplotlib.pyplot as plt
# import torchvision.transforms as T
# from pycocotools.coco import COCO

# # Tambahkan path src agar import aman
# sys.path.append("src")

# try:
#     from src.model import DETR
# except ImportError:
#     from model import DETR

# # === CONFIG ===
# MODEL_PATH = "pretrained/999_model.pt"
# GT_JSON_PATH = "data/test_coco_result.json"
# IMAGE_ROOT = "data/test/images"
# NUM_CLASSES = 26
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Warna: Hijau = Kunci Jawaban (GT), Merah = Prediksi Model
# COLOR_GT = (0, 255, 0) 
# COLOR_PRED = (255, 0, 0)

# def main():
#     # 1. Load Model (Cukup sekali di luar loop)
#     print("🚀 Loading Model...")
#     model = DETR(num_classes=NUM_CLASSES, num_queries=25).to(DEVICE) 
    
#     checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
#     if 'model_state_dict' in checkpoint:
#         model.load_state_dict(checkpoint['model_state_dict'])
#     elif 'model' in checkpoint:
#         model.load_state_dict(checkpoint['model'])
#     else:
#         model.load_state_dict(checkpoint)
        
#     model.eval()
#     print("✅ Model siap.")

#     # 2. Setup Data COCO
#     coco = COCO(GT_JSON_PATH)
#     all_img_ids = coco.getImgIds()
    
#     # --- AMBIL 4 GAMBAR ACAK ---
#     # Pastikan jumlah gambar cukup
#     num_samples = min(4, len(all_img_ids))
#     selected_ids = random.sample(all_img_ids, num_samples)
    
#     print(f"🎲 Mengambil {num_samples} gambar acak ID: {selected_ids}")

#     # 3. Setup Preprocessing (Sama persis dgn Training)
#     preprocess = T.Compose([
#         T.ToPILImage(),
#         T.Resize((224, 224)), 
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
#     ])

#     # 4. Setup Plot (2x2 Grid)
#     fig, axs = plt.subplots(2, 2, figsize=(16, 16))
#     axs = axs.flatten() # Ubah jadi list datar biar gampang di-loop

#     # --- LOOPING 4 GAMBAR ---
#     for i, img_id in enumerate(selected_ids):
#         ax = axs[i]
        
#         # A. Load Info & Gambar
#         img_info = coco.loadImgs(img_id)[0]
#         file_name = os.path.basename(img_info['file_name']) 
#         img_path = os.path.join(IMAGE_ROOT, file_name)

#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"❌ Gagal baca: {file_name}")
#             ax.text(0.5, 0.5, "Image Not Found", ha='center')
#             continue
            
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         h_orig, w_orig, _ = img.shape
#         img_vis = img_rgb.copy() # Copy buat dicoret-coret

#         # B. GAMBAR GROUND TRUTH (HIJAU)
#         ann_ids = coco.getAnnIds(imgIds=img_id)
#         anns = coco.loadAnns(ann_ids)
        
#         for ann in anns:
#             box = ann['bbox']
#             cat_id = ann['category_id']
#             x, y, w, h = map(int, box)
#             cv2.rectangle(img_vis, (x, y), (x+w, y+h), COLOR_GT, 3)
#             cv2.putText(img_vis, f"GT:{cat_id}", (x, y-10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_GT, 2)

#         # C. PREDIKSI MODEL (MERAH)
#         img_tensor = preprocess(img_rgb).unsqueeze(0).to(DEVICE)

#         with torch.no_grad():
#             outputs = model(img_tensor)

#         probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
#         keep = probas.max(-1).values > 0.5 # Threshold Visualisasi
        
#         bboxes_scaled = outputs['pred_boxes'][0, keep]
#         labels = probas[keep].max(-1).indices
#         scores = probas[keep].max(-1).values

#         # Gambar Prediksi
#         if len(bboxes_scaled) == 0:
#             cv2.putText(img_vis, "NO DETECTION", (50, 50), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

#         for box, label, score in zip(bboxes_scaled, labels, scores):
#             # Un-normalize
#             cx, cy, w, h = box.tolist()
#             cx, cy, w, h = cx * w_orig, cy * h_orig, w * w_orig, h * h_orig
#             x = int(cx - 0.5 * w)
#             y = int(cy - 0.5 * h)
#             w_px = int(w)
#             h_px = int(h)
            
#             cv2.rectangle(img_vis, (x, y), (x+w_px, y+h_px), COLOR_PRED, 3)
#             label_text = f"Pred:{label.item()} ({score:.2f})"
#             cv2.putText(img_vis, label_text, (x, y+h_px+25), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_PRED, 2)

#         # D. Tampilkan di Subplot
#         ax.imshow(img_vis)
#         ax.set_title(f"File: {file_name}", fontsize=10)
#         ax.axis('off')

#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     main()