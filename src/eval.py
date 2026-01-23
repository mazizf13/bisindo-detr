import torch
import cv2
import os
import json
import numpy as np
import torchvision.transforms as T
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from model import DETR

# MODEL_PATH = "pretrained/warnet2/noaug/300_model.pt"
# MODEL_PATH = "pretrained/warnet2/aug/300_model.pt"
MODEL_PATH = "pretrained/warnet2/aug-new/200_model.pt"

GT_JSON_PATH = "data/asli/test/result.json"

IMAGE_ROOT = "data/asli/test/images"

NUM_CLASSES = 26
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.5

def main():
    print(f"🚀 Memulai Evaluasi COCO...")
    print(f"📂 Model: {MODEL_PATH}")
    print(f"📂 GT JSON: {GT_JSON_PATH}")

    model = DETR(num_classes=NUM_CLASSES, num_queries=25).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # Handle loading state_dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print("✅ Model berhasil dimuat.")

    coco_gt = COCO(GT_JSON_PATH)
    img_ids = coco_gt.getImgIds()
    print(f"📸 Total Gambar yang akan dites: {len(img_ids)}")

    results_list = []

    print("⏳ Sedang memproses prediksi...")
    
    with torch.no_grad():
        for idx, img_id in enumerate(img_ids):
            # A. Load Info Gambar dari JSON
            img_info = coco_gt.loadImgs(img_id)[0]
            file_name = os.path.basename(img_info['file_name'])
            
            # B. Buka Gambar Asli
            img_path = os.path.join(IMAGE_ROOT, file_name)
            original_img = cv2.imread(img_path)
            
            if original_img is None:
                print(f"⚠️ Gagal baca gambar: {img_path}")
                continue

            img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            h_orig, w_orig, _ = img_rgb.shape

            # Normalize & Tensor
            transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            img_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE) # type: ignore

            outputs = model(img_tensor)

            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1] 
            keep = probas.max(-1).values > CONFIDENCE_THRESHOLD
            
            bboxes_scaled = outputs['pred_boxes'][0, keep]
            probas = probas[keep]

            # F. Konversi Format Box (DETR -> COCO)
            # DETR: (cx, cy, w, h) RELATIF (0-1)
            # COCO: (x_min, y_min, w, h) ABSOLUT (Pixel)
            
            xc, yc, w, h = bboxes_scaled.unbind(-1)
            
            b_x = (xc - 0.5 * w) * w_orig
            b_y = (yc - 0.5 * h) * h_orig
            b_w = w * w_orig
            b_h = h * h_orig
            
            final_boxes = torch.stack([b_x, b_y, b_w, b_h], dim=1).tolist()
            final_scores = probas.max(-1).values.tolist()
            final_labels = probas.max(-1).indices.tolist()

            for box, score, label in zip(final_boxes, final_scores, final_labels):
                results_list.append({
                    "image_id": img_id,       
                    "category_id": label,     
                    "bbox": box,              
                    "score": score
                })

            if idx % 20 == 0:
                print(f"   Proses {idx}/{len(img_ids)} gambar...")

    if len(results_list) == 0:
        print("❌ Tidak ada deteksi sama sekali! Cek model atau preprocessing.")
        return

    pred_json_path = "evaluation_result/black_gpu_temp_predictions.json"
    with open(pred_json_path, "w") as f:
        json.dump(results_list, f)
    
    print(f"✅ Prediksi selesai. Menjalankan COCOeval...")
    
    coco_dt = coco_gt.loadRes(pred_json_path)
    
    # Run Eval
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print("\n🎉 SELESAI! Lihat tabel di atas.")
    print("Baris 'AP @ IoU=0.50' adalah nilai yang kamu cari buat Skripsi.")

if __name__ == "__main__":
    main()


# EVAL DATA OOD
# import os
# import json
# import cv2
# import torch
# import numpy as np
# import torchvision.transforms as T
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# from model import DETR

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# MODEL_PATH = os.path.join(BASE_DIR, "..", "pretrained", "warnet2", "aug-new", "200_model.pt")
# GT_JSON_PATH = os.path.join(BASE_DIR, "..", "data", "lawas-test_ood", "test_ood_result.json")
# IMAGE_ROOT = os.path.join(BASE_DIR, "..", "data", "lawas-test_ood", "images")
# PRED_JSON_PATH = os.path.join(BASE_DIR, "..", "evaluation_result", "black_gpu_temp_predictions.json")

# NUM_CLASSES = 26
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# CONFIDENCE_THRESHOLD = 0.5


# def main():
#     print("🚀 Memulai Evaluasi COCO (OOD)")
#     print(f"📂 Model      : {MODEL_PATH}")
#     print(f"📂 GT JSON    : {GT_JSON_PATH}")
#     print(f"📂 Image Root : {IMAGE_ROOT}")

#     model = DETR(num_classes=NUM_CLASSES, num_queries=25).to(DEVICE)

#     checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
#     if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
#         model.load_state_dict(checkpoint["model_state_dict"])
#     else:
#         model.load_state_dict(checkpoint)

#     model.eval()
#     print("✅ Model berhasil dimuat.")

#     coco_gt = COCO(GT_JSON_PATH)
#     img_ids = coco_gt.getImgIds()
#     print(f"📸 Total gambar: {len(img_ids)}")

#     results_list = []
#     skipped = 0

#     transform = T.Compose([
#         T.ToPILImage(),
#         T.Resize((224, 224)),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406],
#                     [0.229, 0.224, 0.225])
#     ])

#     with torch.no_grad():
#         for idx, img_id in enumerate(img_ids):
#             img_info = coco_gt.loadImgs(img_id)[0]
#             file_name = img_info["file_name"]

#             # Path awal (sesuai JSON)
#             img_path = os.path.normpath(os.path.join(IMAGE_ROOT, file_name))

#             original_img = None

#             # Coba path asli dari JSON
#             if os.path.exists(img_path):
#                 original_img = cv2.imread(img_path)

#             # 🔁 Fallback: buang suffix .rf.xxx jika file tidak ada
#             if original_img is None and ".rf." in file_name:
#                 alt_name = file_name.split(".rf.")[0] + ".jpg"
#                 alt_path = os.path.normpath(os.path.join(IMAGE_ROOT, alt_name))

#                 if os.path.exists(alt_path):
#                     original_img = cv2.imread(alt_path)
#                     img_path = alt_path  # gunakan path alternatif

#             if original_img is None:
#                 print(f"⚠️ Gagal baca gambar: {file_name}")
#                 skipped += 1
#                 continue

#             img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
#             h_orig, w_orig, _ = img_rgb.shape

#             img_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)

#             outputs = model(img_tensor)

#             probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
#             keep = probas.max(-1).values > CONFIDENCE_THRESHOLD

#             if keep.sum() == 0:
#                 continue

#             bboxes_scaled = outputs["pred_boxes"][0, keep]
#             probas = probas[keep]

#             xc, yc, w, h = bboxes_scaled.unbind(-1)

#             b_x = (xc - 0.5 * w) * w_orig
#             b_y = (yc - 0.5 * h) * h_orig
#             b_w = w * w_orig
#             b_h = h * h_orig

#             final_boxes = torch.stack([b_x, b_y, b_w, b_h], dim=1).tolist()
#             final_scores = probas.max(-1).values.tolist()
#             final_labels = probas.max(-1).indices.tolist()

#             for box, score, label in zip(final_boxes, final_scores, final_labels):
#                 results_list.append({
#                     "image_id": img_id,
#                     "category_id": int(label) + 1,
#                     "bbox": box,
#                     "score": float(score)
#                 })

#             if idx % 20 == 0:
#                 print(f"   Proses {idx}/{len(img_ids)} gambar...")


#         print(f"⏭️  Gambar dilewati (gagal dibaca): {skipped}")

#         if len(results_list) == 0:
#             print("❌ Tidak ada deteksi sama sekali! Cek model, threshold, atau preprocessing.")
#             return

#         os.makedirs(os.path.dirname(PRED_JSON_PATH), exist_ok=True)
#         with open(PRED_JSON_PATH, "w") as f:
#             json.dump(results_list, f)

#         print("✅ Prediksi selesai. Menjalankan COCOeval...")

#     coco_dt = coco_gt.loadRes(PRED_JSON_PATH)
#     coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
#     coco_eval.evaluate()
#     coco_eval.accumulate()
#     coco_eval.summarize()

#     print("\n🎉 SELESAI!")
#     print("Gunakan baris 'AP @ IoU=0.50' sebagai hasil utama skripsi.")


# if __name__ == "__main__":
#     main()
