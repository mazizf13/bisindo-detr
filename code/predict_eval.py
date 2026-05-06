import torch
import cv2
import os
import json
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import DETR
import sys

# === CONFIG ===
MODEL_PATH = "pretrained/999_model.pt"
IMAGE_ROOT = "data/test/images"  # Path to raw images
OUTPUT_JSON = "temp_predict_eval.json"
NUM_CLASSES = 26
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the exact same transform as training (Validation mode)
# This ensures the model sees what it expects
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

def main():
    # 1. Load Model
    print(f"🚀 Loading Model from {MODEL_PATH}...")
    model = DETR(num_classes=NUM_CLASSES, num_queries=25).to(DEVICE)
    
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 2. Process Images
    print(f"📂 Processing images in {IMAGE_ROOT}...")
    image_files = [f for f in os.listdir(IMAGE_ROOT) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    results = []
    
    # We need to map filenames to IDs to match your test_coco_result.json
    # Let's load the GT json just to get the ID mapping
    with open("data/test_coco_result.json", 'r') as f:
        gt_data = json.load(f)
    
    # Create a map: filename -> image_id
    # Clean the filename from GT json (remove paths)
    filename_to_id = {}
    for img in gt_data['images']:
        clean_name = os.path.basename(img['file_name'])
        filename_to_id[clean_name] = img['id']

    count = 0
    with torch.no_grad():
        for img_name in image_files:
            if img_name not in filename_to_id:
                continue # Skip images not in GT (if any)
                
            img_id = filename_to_id[img_name]
            img_path = os.path.join(IMAGE_ROOT, img_name)
            
            # Read Image
            original_image = cv2.imread(img_path)
            if original_image is None: continue
            
            # Get Original Dimensions
            h_orig, w_orig, _ = original_image.shape
            
            # Preprocess (Albumentations)
            # We convert BGR to RGB first
            image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            transformed = transform(image=image_rgb)
            img_tensor = transformed['image'].unsqueeze(0).to(DEVICE)

            # Inference
            outputs = model(img_tensor)
            
            # Process Outputs
            probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
            # Low threshold to catch everything for mAP calculation
            keep = probas.max(-1).values > 0.01 
            
            bboxes_scaled = outputs['pred_boxes'][0, keep]
            probs = probas[keep]
            
            # Convert Boxes and Store
            for box, prob in zip(bboxes_scaled, probs):
                score, label = prob.max(-1)
                
                # DETR outputs (cx, cy, w, h) in 0-1 relative coordinates
                cx, cy, w, h = box.tolist()
                
                # Convert to Pixel Coordinates (using ORIGINAL dimensions)
                # This is the crucial step!
                cx *= w_orig
                cy *= h_orig
                w *= w_orig
                h *= h_orig
                
                # Convert Center-Format to Top-Left Format (COCO)
                x_min = cx - (0.5 * w)
                y_min = cy - (0.5 * h)
                
                results.append({
                    "image_id": img_id,
                    "category_id": label.item(),
                    "bbox": [x_min, y_min, w, h],
                    "score": score.item()
                })
            
            count += 1
            if count % 20 == 0:
                print(f"   Processed {count}/{len(image_files)} images")

    # 3. Save Results
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f)
    print(f"✅ Saved {len(results)} predictions to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()