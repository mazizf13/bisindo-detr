import torch
import cv2
import os
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as T

# Tambahkan path src
sys.path.append("src")
try:
    from src.model import DETR
except ImportError:
    from model import DETR

# === CONFIG ===
# Nama file yang gagal tadi (Copy dari terminalmu)
TARGET_FILE = "70ba8b29-M-a1d0429a-942b-11f0-9b15-899b7bd8aca6.jpg"
IMAGE_ROOT = "data/test/images"
MODEL_PATH = "pretrained/999_model.pt"
NUM_CLASSES = 26
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. Load Model
    model = DETR(num_classes=NUM_CLASSES, num_queries=25).to(DEVICE) 
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 2. Load Gambar Spesifik
    img_path = os.path.join(IMAGE_ROOT, TARGET_FILE)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"❌ Gagal baca gambar: {img_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig, _ = img.shape
    img_vis = img_rgb.copy()

    # 3. Preprocessing
    preprocess = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)), 
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
    ])
    
    img_tensor = preprocess(img_rgb).unsqueeze(0).to(DEVICE)

    # 4. Prediksi
    print(f"🔍 Menganalisis {TARGET_FILE}...")
    with torch.no_grad():
        outputs = model(img_tensor)

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    
    # KITA TURUNKAN THRESHOLD KHUSUS BUAT LIHAT APA YANG DILIHAT MODEL
    keep = probas.max(-1).values > 0.1 # Set rendah banget
    
    bboxes_scaled = outputs['pred_boxes'][0, keep]
    labels = probas[keep].max(-1).indices
    scores = probas[keep].max(-1).values

    # 5. Gambar Kotak
    found = False
    for box, label, score in zip(bboxes_scaled, labels, scores):
        # Un-normalize
        cx, cy, w, h = box.tolist()
        cx, cy, w, h = cx * w_orig, cy * h_orig, w * w_orig, h * h_orig
        x = int(cx - 0.5 * w)
        y = int(cy - 0.5 * h)
        w_px = int(w)
        h_px = int(h)
        
        # Huruf M itu index ke-12 (A=0, ..., M=12)
        # Kita cek apakah dia nebak M?
        huruf = chr(ord('A') + label.item())
        
        # Warna: Kuning kalau score rendah, Hijau kalau tinggi
        color = (0, 255, 0) if score > 0.5 else (255, 255, 0)
        
        print(f"   -> Deteksi: Huruf '{huruf}' dengan Score {score:.2f}")
        
        cv2.rectangle(img_vis, (x, y), (x+w_px, y+h_px), color, 3)
        cv2.putText(img_vis, f"{huruf}: {score:.2f}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        found = True

    if not found:
        print("   -> Model benar-benar buta (Score < 0.1)")

    # 6. Tampilkan
    plt.figure(figsize=(8, 8))
    plt.imshow(img_vis)
    plt.title(f"Analisis Kegagalan: {TARGET_FILE}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()