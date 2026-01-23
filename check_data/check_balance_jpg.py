import os

# --- KONFIGURASI ---
base_dir = 'data/black/'
folders = ['train', 'val', 'test']

def check_distribution():
    # Dictionary untuk menyimpan hitungan: data[kelas][folder] = jumlah
    data_counts = {}
    
    print(f"Sedang memeriksa folder di '{base_dir}'...\n")

    # 1. Loop setiap folder (train, val, test)
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        
        # Cek apakah folder ada
        if not os.path.exists(folder_path):
            print(f"[WARNING] Folder '{folder_path}' tidak ditemukan!")
            continue
            
        files = [f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')]
        
        # 2. Hitung file per kelas
        for filename in files:
            # Asumsi nama file: "A-uuid.jpg" -> ambil "A"
            try:
                class_name = filename.split('-')[0]
            except IndexError:
                print(f"[SKIP] Format nama file aneh: {filename}")
                continue
            
            # Inisialisasi jika kelas belum ada di dictionary
            if class_name not in data_counts:
                data_counts[class_name] = {f: 0 for f in folders}
            
            data_counts[class_name][folder] += 1

    # 3. Tampilkan Tabel
    if not data_counts:
        print("Tidak ada data ditemukan.")
        return

    # Header Tabel
    header = f"{'KELAS':<10} | {'TRAIN':<8} | {'VAL':<8} | {'TEST':<8} | {'TOTAL':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # Urutkan berdasarkan abjad (A, B, C...)
    sorted_classes = sorted(data_counts.keys())
    
    total_train = 0
    total_val = 0
    total_test = 0

    for cls in sorted_classes:
        n_train = data_counts[cls].get('train', 0)
        n_val = data_counts[cls].get('val', 0)
        n_test = data_counts[cls].get('test', 0)
        total = n_train + n_val + n_test
        
        total_train += n_train
        total_val += n_val
        total_test += n_test
        
        # Cetak baris
        print(f"{cls:<10} | {n_train:<8} | {n_val:<8} | {n_test:<8} | {total:<8}")

    print("-" * len(header))
    print(f"{'ALL':<10} | {total_train:<8} | {total_val:<8} | {total_test:<8} | {total_train+total_val+total_test:<8}")

if __name__ == "__main__":
    check_distribution()