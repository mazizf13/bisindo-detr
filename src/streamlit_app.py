# ------------------------- DEPLOYMENT CODE -------------------------
import streamlit as st
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import DETR
from utils.setup import get_classes, get_colors
from utils.boxes import rescale_bboxes
import numpy as np
import time

# -------------------------
# Setup
# -------------------------
st.title("📸 Real-time Alfabet BISINDO (Streamlit)")

# Detect device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
st.text(f"Running on device: {device}")

# Load model
model = DETR(num_classes=26)
model = model.to(device)
model.eval()
model.load_pretrained('model/300_model.pt', map_location=device)

CLASSES = get_classes()
COLORS = get_colors()

# Transforms
transforms = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ToTensorV2()
])

# -------------------------
# Streamlit UI - Camera Input
# -------------------------
st.subheader("📸 Ambil gambar dari kamera atau upload file")

img_file = st.camera_input("Ambil gambar dari kamera")

if img_file is not None:
    # Convert file ke OpenCV image
    bytes_data = img_file.read()
    np_img = np.frombuffer(bytes_data, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Mirror
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape

    # Preprocess
    transformed = transforms(image=frame)
    img_tensor = torch.unsqueeze(transformed['image'], dim=0).to(device)

    # Inference
    with torch.no_grad():
        start_time = time.time()
        result = model(img_tensor)
        inference_time = (time.time() - start_time) * 1000  # ms

    # Post-processing: softmax + Top-1 threshold
    probabilities = result['pred_logits'].softmax(-1)[0,:,:-1].cpu()
    max_probs, max_classes = probabilities.max(-1)
    top_score, top_idx = max_probs.max(0)

    if top_score > 0.7:
        pred_boxes = result['pred_boxes'][0, top_idx].unsqueeze(0).cpu()
        keep_class = max_classes[top_idx].item()
        keep_prob = max_probs[top_idx].item()

        bboxes = rescale_bboxes(pred_boxes, (width, height))

        for bbox in bboxes:
            x1, y1, x2, y2 = map(int, bbox.numpy())

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[keep_class], 3)
            # Draw label
            label = f"{CLASSES[keep_class]} {keep_prob:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 5), (x1 + text_w, y1), COLORS[keep_class], -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Tampilkan hasil
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
             caption=f"Hasil Deteksi | inference {inference_time:.1f} ms",
             use_container_width=True)



# ------------------------- LOCAL TESTING ONLY -------------------------
# import streamlit as st
# import cv2
# import torch
# import time
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from model import DETR
# from utils.setup import get_classes, get_colors
# from utils.boxes import rescale_bboxes
# import numpy as np

# # -------------------------
# # Setup
# # -------------------------
# st.title("Real-time Sign Language Detection (Streamlit)")

# # Detect device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# st.text(f"Running on device: {device}")

# # Load model
# model = DETR(num_classes=26)
# model = model.to(device)
# model.eval()
# model.load_pretrained('model/300_model.pt', map_location=device)

# CLASSES = get_classes()
# COLORS = get_colors()

# # Transforms
# transforms = A.Compose([
#     A.Resize(224,224),
#     A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
#     ToTensorV2()
# ])

# # -------------------------
# # Streamlit UI
# # -------------------------
# run = st.checkbox("Start Camera")
# FRAME_WINDOW = st.image([])
# fps_text = st.empty()

# cap = cv2.VideoCapture(0)

# # FPS tracking
# frame_count = 0
# fps_start_time = time.time()
# fps_display = 0.0

# try:
#     with torch.no_grad():
#         while run:
#             ret, frame = cap.read()
#             if not ret:
#                 st.error("Failed to read frame from camera")
#                 break

#             frame = cv2.flip(frame, 1)  # Mirror

#             # Preprocess
#             transformed = transforms(image=frame)
#             img_tensor = torch.unsqueeze(transformed['image'], dim=0).to(device)

#             # Inference
#             inference_start = time.time()
#             result = model(img_tensor)
#             inference_time = (time.time() - inference_start) * 1000  # ms

#             # Post-processing: softmax + Top-1 threshold
#             probabilities = result['pred_logits'].softmax(-1)[0,:,:-1].detach().cpu()
#             max_probs, max_classes = probabilities.max(-1)

#             top_score, top_idx = max_probs.max(0)
#             if top_score > 0.7:
#                 query_indices = top_idx.unsqueeze(0)
#             else:
#                 query_indices = []

#             height, width, _ = frame.shape

#             if len(query_indices) > 0:
#                 pred_boxes = result['pred_boxes'][0, query_indices].detach().cpu()
#                 keep_classes = max_classes[query_indices].detach().cpu()
#                 keep_probs = max_probs[query_indices].detach().cpu()
#                 bboxes = rescale_bboxes(pred_boxes, (width, height))

#                 for bclass, bprob, bbox in zip(keep_classes, keep_probs, bboxes):
#                     x1, y1, x2, y2 = map(int, bbox.numpy())
#                     bclass_idx = int(bclass.numpy())
#                     bprob_val = float(bprob.numpy())

#                     # Draw rectangle
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), COLORS[bclass_idx], 3)
#                     # Draw label background
#                     (text_w, text_h), baseline = cv2.getTextSize(f"{CLASSES[bclass_idx]} {bprob_val:.2f}",
#                                                                   cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
#                     cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), COLORS[bclass_idx], -1)
#                     # Put text
#                     cv2.putText(frame, f"{CLASSES[bclass_idx]} {bprob_val:.2f}", (x1, y1 - 5),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

#             # FPS update
#             frame_count += 1
#             if frame_count % 5 == 0:
#                 elapsed_time = time.time() - fps_start_time
#                 fps_display = 5 / elapsed_time if elapsed_time > 0 else 0
#                 fps_start_time = time.time()

#             # Overlay FPS
#             cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#             # Display
#             FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             fps_text.text(f"FPS: {fps_display:.1f} | Device: {device}")

# except KeyboardInterrupt:
#     st.warning("Interrupted by user")

# finally:
#     cap.release()


# # import streamlit as st
# # import cv2
# # import torch
# # import albumentations as A
# # from model import DETR
# # from utils.setup import get_classes, get_colors
# # from utils.boxes import rescale_bboxes
# # import numpy as np

# # # Load model
# # model = DETR(num_classes=26)
# # model.eval()
# # model.load_pretrained('pretrained/warnet2/aug-new/300_model.pt', map_location=torch.device('cpu'))
# # CLASSES = get_classes()
# # COLORS = get_colors()

# # # Transforms
# # transforms = A.Compose([
# #     A.Resize(224,224),
# #     A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
# #     A.ToTensorV2()
# # ])

# # st.title("Real-time Sign Language Detection")

# # run = st.checkbox("Start Camera")

# # FRAME_WINDOW = st.image([])

# # cap = cv2.VideoCapture(0)

# # while run:
# #     ret, frame = cap.read()
# #     frame = cv2.flip(frame, 1)

# #     transformed = transforms(image=frame)
# #     result = model(torch.unsqueeze(transformed['image'], dim=0))

# #     probabilities = result['pred_logits'].softmax(-1)[:,:,:-1]
# #     max_probs, max_classes = probabilities.max(-1)
# #     keep_mask = max_probs > 0.8

# #     batch_indices, query_indices = torch.where(keep_mask)
# #     height, width, _ = frame.shape
# #     bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices,:], (width, height))
# #     classes = max_classes[batch_indices, query_indices]
# #     probas = max_probs[batch_indices, query_indices]

# #     for bclass, bprob, bbox in zip(classes, probas, bboxes):
# #         x1,y1,x2,y2 = bbox.detach().numpy()
# #         cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), COLORS[int(bclass)], 3)
# #         cv2.putText(frame, f"{CLASSES[int(bclass)]} {bprob:.2f}", 
# #                     (int(x1),int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# #     FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# # cap.release()


