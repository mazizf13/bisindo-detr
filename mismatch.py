import os
import re

# folder tempat gambar Roboflow
image_folder = "data/test_ood/labels"

# list semua file
files = os.listdir(image_folder)

for file_name in files:
    # cek kalau ada pattern .rf.[hash] sebelum ekstensi
    new_name = re.sub(r'\.rf\.[a-f0-9]+', '', file_name)
    if new_name != file_name:
        old_path = os.path.join(image_folder, file_name)
        new_path = os.path.join(image_folder, new_name)
        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} -> {new_name}")

print("Rename selesai!")
