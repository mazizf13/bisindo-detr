# PART 2 ==========================================================================================
import os
import sys

os.environ["PYTHONIOENCODING"] = "utf-8"
try:
    sys.stdout.reconfigure(encoding='utf-8') # type: ignore
except AttributeError:
    pass

import cv2
import torch
import time
import albumentations as A
from model import DETR
from utils.boxes import rescale_bboxes
from utils.setup import get_classes, get_colors
from utils.logger import get_logger
from utils.rich_handlers import DetectionHandler
from albumentations.pytorch import ToTensorV2

# Logger
logger = get_logger("realtime")
detection_handler = DetectionHandler()

logger.print_banner()
logger.realtime("Initializing real-time sign language detection...")

# Preprocessing pipeline
transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# 1. AUTO DETECT DEVICE (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.realtime(f"Running inference on: {device}") 

# Load model
model = DETR(num_classes=26)
model = model.to(device)
model.eval()

try:
    # 2. LOAD PRETRAINED
    # model.load_pretrained('pretrained/warnet/rabu/blacknoaug/300_model.pt', map_location=device)
    # model.load_pretrained('pretrained/warnet2/noaug/300_model.pt', map_location=device)
    # model.load_pretrained('pretrained/warnet2/aug/200_model.pt', map_location=device)
    model.load_pretrained('pretrained/warnet2/aug-new/300_model.pt', map_location=device)
    # model.load_pretrained('pretrained/warnet/gpu_black_300_model.pt', map_location=device)
    try:
        logger.success("Model loaded successfully!")
    except:
        print("[INFO] Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    sys.exit(1)

CLASSES = get_classes()
COLORS = get_colors()

logger.realtime("Starting camera capture...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Failed to open camera")
    sys.exit(1)

# Initialize performance tracking
frame_count = 0
fps_start_time = time.time()
session_start_time = time.time()
fps_display = 0.0 

# --- FIX 2: WRAP DENGAN TORCH.NO_GRAD() ---
try:
    with torch.no_grad():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame from camera")
                break

            # Mirror effect
            frame = cv2.flip(frame, 1)

            # Inference Setup
            inference_start = time.time()
            
            # Preprocess
            transformed = transforms(image=frame)
            # Pindah gambar ke GPU/CPU
            img_tensor = torch.unsqueeze(transformed['image'], dim=0).to(device)
            
            # Forward Pass
            result = model(img_tensor)
            
            inference_time = (time.time() - inference_start) * 1000  # ms

            # --- POST PROCESSING (SAFE MODE) ---
            probabilities = result['pred_logits'].softmax(-1)[0, :, :-1].detach().cpu()
            max_probs, max_classes = probabilities.max(-1)
            
            # keep_mask = max_probs > 0.7
            # query_indices = keep_mask.nonzero(as_tuple=True)[0]
            # --- BAGIAN YANG DIUBAH (Top-1 Only) ---
            # 1. Cari 1 score tertinggi dan index-nya dari 100 queries
            top_score, top_idx = max_probs.max(0) 
            
            # 2. Set threshold rendah (misal 0.1) biar box tetap muncul walau confidence turun,
            # tapi tidak muncul kalau gelap total/noise parah.
            if top_score > 0.7:
                # .unsqueeze(0) penting biar formatnya jadi list [index], bukan angka tunggal
                # jadi kode di bawahnya (for loop) tetap jalan normal.
                query_indices = top_idx.unsqueeze(0)
            else:
                query_indices = []
            # --- SELESAI UBAH ---

            # Siapkan list detections untuk Logger & Drawing
            current_detections = []

            if len(query_indices) > 0:
                # Ambil boxes, detach, pindah cpu
                pred_boxes = result['pred_boxes'][0, query_indices].detach().cpu()
                keep_classes = max_classes[query_indices].detach().cpu()
                keep_probs = max_probs[query_indices].detach().cpu()

                height, width, _ = frame.shape
                bboxes = rescale_bboxes(pred_boxes, (width, height))

                for bclass, bprob, bbox in zip(keep_classes, keep_probs, bboxes):
                    # Convert ke numpy aman karena sudah detach().cpu()
                    bclass_idx = int(bclass.numpy())
                    bprob_val = float(bprob.numpy())
                    x1, y1, x2, y2 = map(int, bbox.numpy())

                    if bclass_idx >= len(CLASSES):
                        continue

                    # Logger Terminal
                    current_detections.append({
                        'class': CLASSES[bclass_idx],
                        'confidence': bprob_val,
                        'bbox': [x1, y1, x2, y2]
                    })

                    # Draw bounding box di layar
                    cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[bclass_idx], 3) # type: ignore

                    # Label text
                    frame_text = f"{CLASSES[bclass_idx]} - {bprob_val:.2f}"
                    (text_w, text_h), baseline = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)

                    cv2.rectangle(
                        frame,
                        (x1, y1 - text_h - 10),
                        (x1 + text_w + 10, y1),
                        COLORS[bclass_idx],
                        -1
                    ) # type: ignore

                    cv2.putText(
                        frame,
                        frame_text,
                        (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1.2,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

            # LOGGING TERMINAL
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - fps_start_time
                fps_display = 30 / elapsed_time if elapsed_time > 0 else 0
                
                # 1. Log Tabel Deteksi
                if current_detections:
                    try:
                        detection_handler.log_detections(current_detections, frame_id=frame_count)
                    except:
                        pass # Skip kalau error encoding tabel

                # 2. Log Latency & FPS
                try:
                    detection_handler.log_inference_time(inference_time, fps_display)
                except:
                    print(f"[FPS: {fps_display:.2f} | Latency: {inference_time:.2f}ms]") # Fallback manual

                fps_start_time = time.time()
                
            # Display FPS & Device di Layar Kamera
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Device: {device}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow('Sign Language Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # kalao error emoji lagi, try-except
                try:
                    logger.realtime("Stopping real-time detection...")
                except:
                    pass
                break
except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user (Ctrl+C). Exiting safely...")

# Rata-rata Akhir 
total_session_time = time.time() - session_start_time
final_avg_fps = frame_count / total_session_time if total_session_time > 0 else 0

cap.release()
cv2.destroyAllWindows()

# Final Report 
print("\n" + "="*40)
print(f"✅ Session Ended Successfully")
print(f"📊 Total Frames: {frame_count}")
print(f"⏱️ Total Duration: {total_session_time:.2f}s")
print(f"🚀 FINAL AVERAGE FPS: {final_avg_fps:.2f}") 
print(f"💻 Hardware Used: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print("="*40 + "\n")

# PAKAI YANG PART 2
# PART 3 with ccalcualte fps and latency ==========================================================================================
# import cv2
# import torch
# import csv
# import time
# import statistics
# import sys
# import albumentations as A
# from torch import load
# from model import DETR
# from utils.boxes import rescale_bboxes
# from utils.setup import get_classes, get_colors
# from utils.logger import get_logger
# from utils.rich_handlers import DetectionHandler
# from albumentations.pytorch import ToTensorV2

# # Logger
# logger = get_logger("realtime")
# detection_handler = DetectionHandler()

# logger.print_banner()
# logger.realtime("Initializing real-time sign language detection...")

# # Preprocessing pipeline
# transforms = A.Compose([
#     A.Resize(224, 224),
#     A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ToTensorV2()
# ])

# # Load model
# model = DETR(num_classes=26)
# model.eval()
# try:
#     model.load_pretrained('pretrained/150_final_model.pt', map_location=torch.device('cpu'))
#     logger.success("Model loaded successfully!")
# except Exception as e:
#     logger.error(f"Failed to load model: {e}")
#     sys.exit(1)

# CLASSES = get_classes()
# COLORS = get_colors()

# logger.realtime("Starting camera capture...")
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     logger.error("Failed to open camera")
#     sys.exit(1)

# # Initialize performance tracking
# frame_count = 0
# fps_start_time = time.time()

# fps_values = []
# latency_values = []

# # Prepare CSV
# csv_file = open("performance_log.csv", mode="w", newline="")
# csv_writer = csv.writer(csv_file)
# csv_writer.writerow(["Frame_ID", "FPS", "Latency_ms"])

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         logger.error("Failed to read frame from camera")
#         break

#     frame = cv2.flip(frame, 1)

#     # Inference
#     inference_start = time.time()
#     transformed = transforms(image=frame)
#     result = model(torch.unsqueeze(transformed['image'], dim=0))
#     inference_time = (time.time() - inference_start) * 1000  # ms
#     latency_values.append(inference_time)

#     probabilities = result['pred_logits'].softmax(-1)[:, :, :-1]
#     max_probs, max_classes = probabilities.max(-1)
#     keep_mask = max_probs > 0.5

#     batch_indices, query_indices = torch.where(keep_mask)

#     height, width, _ = frame.shape
#     bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices, :], (width, height))
#     classes = max_classes[batch_indices, query_indices]
#     probas = max_probs[batch_indices, query_indices]

#     detections = []
#     for bclass, bprob, bbox in zip(classes, probas, bboxes):
#         bclass_idx = int(bclass.detach().numpy())
#         bprob_val = float(bprob.detach().numpy())
#         x1, y1, x2, y2 = map(int, bbox.detach().numpy())

#         if bclass_idx >= len(CLASSES):
#             continue

#         detections.append({
#             'class': CLASSES[bclass_idx],
#             'confidence': bprob_val,
#             'bbox': [x1, y1, x2, y2]
#         })

#         # Draw
#         cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[bclass_idx], 3)
#         frame_text = f"{CLASSES[bclass_idx]} - {bprob_val:.4f}"
#         (text_w, text_h), baseline = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 2)
#         cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), COLORS[bclass_idx], -1)
#         cv2.putText(frame, frame_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

#     # FPS calculation
#     frame_count += 1
#     if frame_count % 30 == 0:
#         elapsed_time = time.time() - fps_start_time
#         fps = 30 / elapsed_time if elapsed_time > 0 else 0
#         fps_values.append(fps)

#         if detections:
#             detection_handler.log_detections(detections, frame_id=frame_count)
#         detection_handler.log_inference_time(inference_time, fps)

#         # Save to CSV
#         csv_writer.writerow([frame_count, f"{fps:.2f}", f"{inference_time:.2f}"])

#         logger.realtime(f"[Frame {frame_count}] FPS: {fps:.2f} | Latency: {inference_time:.2f} ms")

#         fps_start_time = time.time()

#     cv2.imshow('Sign Language Detection', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         logger.realtime("Stopping real-time detection...")
#         break

# # End session
# csv_file.close()
# cap.release()
# cv2.destroyAllWindows()

# # Compute averages
# if fps_values and latency_values:
#     avg_fps = statistics.mean(fps_values)
#     avg_latency = statistics.mean(latency_values)
#     logger.success(f"Average FPS: {avg_fps:.2f}")
#     logger.success(f"Average Latency: {avg_latency:.2f} ms")
#     logger.info(f"Min FPS: {min(fps_values):.2f} | Max FPS: {max(fps_values):.2f}")
#     logger.info(f"Min Latency: {min(latency_values):.2f} ms | Max Latency: {max(latency_values):.2f} ms")

# PART 1 kebesaran ==========================================================================================
# import cv2
# import torch
# from torch import load
# from model import DETR
# import albumentations as A
# from utils.boxes import rescale_bboxes
# from utils.setup import get_classes, get_colors
# from utils.logger import get_logger
# from utils.rich_handlers import DetectionHandler, create_detection_live_display
# import sys
# import time 


# # Logger
# logger = get_logger("realtime")
# detection_handler = DetectionHandler()

# logger.print_banner()
# logger.realtime("Initializing real-time sign language detection...")

# transforms = A.Compose(
#         [   
#             A.Resize(224,224),
#             A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             A.ToTensorV2()
#         ]
#     )

# model = DETR(num_classes=26)
# model.eval()
# model.load_pretrained('pretrained/cpu_300_final_model.pt', map_location=torch.device('cpu'))
# CLASSES = get_classes() 
# COLORS = get_colors() 

# logger.realtime("Starting camera capture...")
# cap = cv2.VideoCapture(0)

# # Initialize performance tracking
# frame_count = 0
# fps_start_time = time.time()

# while cap.isOpened(): 
#     ret, frame = cap.read()
#     if not ret:
#         logger.error("Failed to read frame from camera")
#         break

#     # Mirror
#     frame = cv2.flip(frame, 1)
        
#     # Time the inference
#     inference_start = time.time()
#     transformed = transforms(image=frame)
#     result = model(torch.unsqueeze(transformed['image'], dim=0))
#     inference_time = (time.time() - inference_start) * 1000  # Convert to ms

#     probabilities = result['pred_logits'].softmax(-1)[:,:,:-1] 
#     max_probs, max_classes = probabilities.max(-1)
#     keep_mask = max_probs > 0.8

#     batch_indices, query_indices = torch.where(keep_mask) 

#     height, width, _ = frame.shape
#     bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices,:], (width, height))
#     # bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices,:], (1920,1080))
#     classes = max_classes[batch_indices, query_indices]
#     probas = max_probs[batch_indices, query_indices]

#     # Prepare detection results for logging
#     detections = []
#     for bclass, bprob, bbox in zip(classes, probas, bboxes): 
#         bclass_idx = bclass.detach().numpy()
#         bprob_val = bprob.detach().numpy() 
#         x1,y1,x2,y2 = bbox.detach().numpy()
        
#         detections.append({
#             'class': CLASSES[bclass_idx],
#             'confidence': float(bprob_val),
#             'bbox': [float(x1), float(y1), float(x2), float(y2)]
#         })
        
#         # Draw bounding boxes on frame
#         frame = cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), COLORS[bclass_idx], 10)
#         frame_text = f"{CLASSES[bclass_idx]} - {round(float(bprob_val),4)}"
#         frame = cv2.rectangle(frame, (int(x1),int(y1)-100), (int(x1)+700,int(y1)), COLORS[bclass_idx], -1)
#         frame = cv2.putText(frame, frame_text, (int(x1),int(y1)), cv2.FONT_HERSHEY_DUPLEX, 2, (255,255,255), 4, cv2.LINE_AA)

#     # Calculate FPS 
#     frame_count += 1
#     if frame_count % 30 == 0:  # Log every 30 frames
#         elapsed_time = time.time() - fps_start_time
#         fps = 30 / elapsed_time
        
#         # Log detection results and performance
#         if detections:
#             detection_handler.log_detections(detections, frame_id=frame_count)
#         detection_handler.log_inference_time(inference_time, fps)
        
#         # Reset FPS counter
#         fps_start_time = time.time()

#     cv2.imshow('Frame', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'): 
#         logger.realtime("Stopping real-time detection...")
#         break

# cap.release() 
# cv2.destroyAllWindows() 
