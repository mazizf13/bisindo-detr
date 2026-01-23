import torch
from torch.utils.data import DataLoader 
from matplotlib import pyplot as plt 
import time

from data import DETRData
from model import DETR
from utils.boxes import rescale_bboxes
from utils.setup import get_classes
from utils.logger import get_logger
from utils.rich_handlers import TestHandler, DetectionHandler

# Logger & handler
logger = get_logger("test")
test_handler = TestHandler()
detection_handler = DetectionHandler()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.print_banner()

# Dataset & dataloader
num_classes = 26
# test_dataset = DETRData('data/lawas-test_ood', train=False)
test_dataset = DETRData('data/asli/test', train=False)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=8, drop_last=True) 

# Load model
model = DETR(num_classes=num_classes)
model.eval()
model.load_pretrained('pretrained/warnet2/noaug/300_model.pt', map_location=device)
# model.load_pretrained('pretrained/warnet2/aug/200_model.pt', map_location=device)
# model.load_pretrained('pretrained/warnet2/aug-new/200_model.pt', map_location=device)

# Ambil 1 batch data
X, y = next(iter(test_dataloader))
logger.test("Running inference on test batch...")

# Inference
start_time = time.time()
result = model(X) 
inference_time = (time.time() - start_time) * 1000  # ms

# Softmax + threshold
probabilities = result['pred_logits'].softmax(-1)[:, :, :-1] 
max_probs, max_classes = probabilities.max(-1)
keep_mask = max_probs > 0.7
batch_indices, query_indices = torch.where(keep_mask)

# Scale boxes
bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices, :], (224, 224))
classes = max_classes[batch_indices, query_indices]
probas = max_probs[batch_indices, query_indices]

# Log inference time
detection_handler.log_inference_time(inference_time)

# Ambil hanya 1 deteksi tertinggi per image
detections = []
unique_batch_indices = batch_indices.unique()
for b_idx in unique_batch_indices:
    mask = batch_indices == b_idx
    if mask.sum() == 0:
        continue

    scores = probas[mask]
    boxes = bboxes[mask]
    labels = classes[mask]

    top_idx = scores.argmax()

    detections.append({
        'batch': int(b_idx.item()),
        'class': get_classes()[labels[top_idx].item()],
        'confidence': scores[top_idx].item(),
        'bbox': boxes[top_idx].detach().numpy().tolist()
    })

# Log detection results
detection_handler.log_detections(detections)

# Visualisasi
CLASSES = get_classes()
fig, ax = plt.subplots(2, 4)
axs = ax.flatten()

for idx, (img, ax) in enumerate(zip(X, axs)):
    ax.imshow(img.permute(1, 2, 0))
    for det in detections:
        if det['batch'] == idx:
            xmin, ymin, xmax, ymax = det['bbox']
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, # type: ignore
                                       fill=False, color=(0.000, 0.447, 0.741), linewidth=3))
            text = f"{det['class']}: {det['confidence']:.2f}"
            ax.text(xmin, ymin, text, fontsize=12,
                    bbox=dict(facecolor='yellow', alpha=0.5))

fig.tight_layout()
plt.show()

# from data import DETRData
# from model import DETR
# import torch
# from torch import load
# from torch.utils.data import DataLoader 
# from matplotlib import pyplot as plt 
# from utils.boxes import rescale_bboxes
# from utils.setup import get_classes
# from utils.logger import get_logger
# from utils.rich_handlers import TestHandler, DetectionHandler

# # Initialize loggers and handlers
# logger = get_logger("test")
# test_handler = TestHandler()
# detection_handler = DetectionHandler()

# logger.print_banner()

# num_classes = 26
# test_dataset = DETRData('data/test', train=False) 
# test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=4, drop_last=True) 
# model = DETR(num_classes=num_classes)
# model.eval()
# model.load_pretrained('pretrained/300_final_model.pt', map_location=torch.device('cpu'))
# # model.load_pretrained('pretrained/999_model.pt')

# X, y = next(iter(test_dataloader))

# logger.test("Running inference on test batch...")

# import time
# start_time = time.time()
# result = model(X) 
# inference_time = (time.time() - start_time) * 1000  # Convert to ms

# probabilities = result['pred_logits'].softmax(-1)[:,:,:-1] 
# max_probs, max_classes = probabilities.max(-1)
# keep_mask = max_probs > 0.6
# batch_indices, query_indices = torch.where(keep_mask) 

# bboxes = rescale_bboxes(result['pred_boxes'][batch_indices, query_indices,:], (224,224))
# classes = max_classes[batch_indices, query_indices]
# probas = max_probs[batch_indices, query_indices]

# # Log inference timing
# detection_handler.log_inference_time(inference_time)

# # Prepare detection results for logging
# detections = []
# for i in range(len(classes)):
#     detections.append({
#         'class': get_classes()[classes[i].item()],
#         'confidence': probas[i].item(),
#         'bbox': bboxes[i].detach().numpy().tolist()
#     })

# # Log detection results
# detection_handler.log_detections(detections) 

# CLASSES = get_classes()

# fig, ax = plt.subplots(2,2) 
# axs = ax.flatten()
# for idx, (img, ax) in enumerate(zip(X, axs)): 
#     ax.imshow(img.permute(1,2,0))
#     for batch_idx, box_class, box_prob, bbox in zip(batch_indices, classes, probas, bboxes): 
#         if batch_idx == idx: 
#             xmin, ymin, xmax, ymax = bbox.detach().numpy()
#             print(xmin, ymin, xmax, ymax) 
#             ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=(0.000, 0.447, 0.741), linewidth=3))
#             text = f'{CLASSES[box_class]}: {box_prob:0.2f}'
#             ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))

# fig.tight_layout() 
# plt.show()     