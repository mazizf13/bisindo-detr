import cv2
import matplotlib.pyplot as plt
import albumentations as A
import os
import random
import numpy as np

# === CONFIG ===
# Ganti dengan salah satu path gambar di datasetmu
IMAGE_PATH = "data/train/images/0a264e0e-A-37b62f69-9426-11f0-a78c-899b7bd8aca6.jpg"
# Kalau malas cari nama file, biarkan None biar script cari sendiri
# IMAGE_PATH = None 
DATA_DIR = "data/train/images"

def get_random_image():
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
    return os.path.join(DATA_DIR, random.choice(files))

def main():
    # 1. Load Gambar
    img_path = IMAGE_PATH if IMAGE_PATH else get_random_image()
    if not os.path.exists(img_path):
        print(f"❌ Gambar tidak ditemukan: {img_path}")
        return
        
    # Baca gambar dengan OpenCV (BGR) -> Convert ke RGB
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Definisi Transformasi (Sesuai data.py kamu)
    # A. Original (Cuma Resize biar seragam tampilannya)
    transform_original = A.Resize(500, 500)
    
    # B. Random Crop (Simulasi Zoom)
    transform_crop = A.Compose([
        A.Resize(500, 500),
        A.RandomCrop(width=300, height=300, p=1.0), # Crop ekstrem biar kelihatan bedanya
        A.Resize(500, 500) # Balikin ukuran biar grid rapi
    ])
    
    # C. Color Jitter (Simulasi Beda Cahaya)
    transform_color = A.Compose([
        A.Resize(500, 500),
        A.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=1.0) # Nilai ekstrem biar visual jelas
    ])
    
    # D. Horizontal Flip (Simulasi Cermin)
    transform_flip = A.Compose([
        A.Resize(500, 500),
        A.HorizontalFlip(p=1.0)
    ])

    # 3. Terapkan Transformasi
    img_orig = transform_original(image=image)['image']
    img_crop = transform_crop(image=image)['image']
    img_color = transform_color(image=image)['image']
    img_flip = transform_flip(image=image)['image']

    # 4. Plotting Grid 1x4
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Judul dan Gambar
    titles = ["(a) Citra Asli", "(b) Random Crop (Zoom)", "(c) Color Jitter", "(d) Horizontal Flip"]
    images = [img_orig, img_crop, img_color, img_flip]
    
    for ax, img, title in zip(axs, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, pad=10)
        ax.axis('off')
        
        # Tambah border tipis biar kelihatan batasnya (opsional)
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(2)

    plt.tight_layout()
    plt.savefig("contoh_augmentasi.png", dpi=300, bbox_inches='tight')
    print("✅ Gambar tersimpan: contoh_augmentasi.png")
    plt.show()

if __name__ == "__main__":
    main()