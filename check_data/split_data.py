import os
import shutil
import random
from collections import defaultdict

# --- KONFIGURASI ---
# Pastikan path ini sesuai dengan struktur foldermu
source_dir = 'data/black'  # Folder asal (dimana semua 60 gambar berada)
val_dir = 'data/val'       # Folder tujuan validasi
test_dir = 'data/test'     # Folder tujuan testing

# Jumlah yang ingin dipindah
NUM_TEST = 5
NUM_VAL = 7
# Sisanya (48) akan tetap di folder train

# --- EKSEKUSI ---

def split_dataset():
    # 1. Buat folder tujuan jika belum ada
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 2. Dapatkan semua file jpg
    all_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.jpg')]
    
    # 3. Kelompokkan file berdasarkan Kelas (Huruf depan sebelum tanda strip '-')
    files_by_class = defaultdict(list)
    for filename in all_files:
        # Asumsi format: "A-uuid.jpg" -> mengambil "A"
        class_name = filename.split('-')[0]
        files_by_class[class_name].append(filename)

    print(f"Ditemukan {len(files_by_class)} kelas berbeda.")

    # 4. Proses pemindahan per kelas
    for class_name, files in files_by_class.items():
        # Cek apakah jumlah file cukup
        if len(files) < (NUM_TEST + NUM_VAL):
            print(f"Peringatan: Kelas {class_name} hanya punya {len(files)} file. Melewati kelas ini.")
            continue

        # Acak urutan file supaya random
        random.shuffle(files)

        # Tentukan file mana yang dipindah
        files_to_test = files[:NUM_TEST]
        files_to_val = files[NUM_TEST : NUM_TEST + NUM_VAL]
        # Sisanya (files[NUM_TEST + NUM_VAL:]) dibiarkan di folder train
        
        # Pindahkan ke Test
        for f in files_to_test:
            src = os.path.join(source_dir, f)
            dst = os.path.join(test_dir, f)
            shutil.move(src, dst)

        # Pindahkan ke Val
        for f in files_to_val:
            src = os.path.join(source_dir, f)
            dst = os.path.join(val_dir, f)
            shutil.move(src, dst)
        
        print(f"Kelas {class_name}: Dipindah 5 ke test, 7 ke val, sisa {len(files) - NUM_TEST - NUM_VAL} di data black.")

    print("\nSelesai! Data berhasil dibagi.")

if __name__ == "__main__":
    # Pastikan folder data/train ada sebelum dijalankan
    if os.path.exists(source_dir):
        split_dataset()
    else:
        print(f"Error: Folder sumber '{source_dir}' tidak ditemukan.")