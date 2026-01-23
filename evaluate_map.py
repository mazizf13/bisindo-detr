import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
from model import DETR  # ganti sesuai modelmu

# === Fungsi evaluasi COCO ===
def evaluate_coco(model, dataloader, device, coco_gt_json, output_json="predictions.json"):
    model.eval()
    results = []

    # load COCO object untuk ambil info width/height
    coco = COCO(coco_gt_json)

    with torch.no_grad():
        for images, targets in dataloader:
            # gabungkan batch jadi tensor
            images = torch.stack(images).to(device)
            outputs = model(images)

            # normalisasi output jadi list of dict per image
            if isinstance(outputs, dict):
                outputs = [{k: v[i] for k, v in outputs.items()} for i in range(len(images))]
            elif isinstance(outputs, tuple) and len(outputs) == 2:
                pred_logits, pred_boxes = outputs
                outputs = [{"pred_logits": pred_logits[i], "pred_boxes": pred_boxes[i]} for i in range(len(images))]
            elif isinstance(outputs, list):
                pass
            else:
                raise TypeError(f"Unexpected model output type: {type(outputs)}")

            # loop setiap image dalam batch
            for i, output in enumerate(outputs):
                boxes = output["pred_boxes"].cpu()
                scores = output["pred_logits"].softmax(-1).cpu()
                labels = scores.argmax(-1)

                # ambil image_id dari targets
                image_id = targets[i][0]["image_id"] if len(targets[i]) > 0 else -1
                img_info = coco.imgs[image_id]
                img_w = img_info["width"]
                img_h = img_info["height"]

                for box, score, label in zip(boxes, scores.max(-1).values, labels):
                    # skip "no object" class (kelas terakhir)
                    if label.item() == (scores.shape[-1] - 1):
                        continue

                    x_c, y_c, w, h = box.tolist()
                    # convert bbox normalized center -> pixel [x_min, y_min, w, h]
                    coco_box = [
                        (x_c - w/2) * img_w,
                        (y_c - h/2) * img_h,
                        w * img_w,
                        h * img_h,
                    ]   

                    pred = {
                        "image_id": int(image_id),
                        "category_id": int(label.item()),
                        "bbox": coco_box,
                        "score": float(score.item())
                    }
                    results.append(pred)
                    print("DEBUG PRED:", pred)

    # simpan prediksi
    with open(output_json, "w") as f:
        json.dump(results, f)

    # evaluasi COCO
    coco_dt = coco.loadRes(output_json)
    coco_eval = COCOeval(coco, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval.stats

# === Main ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = DETR(num_classes=26)
    model.load_state_dict(torch.load("pretrained/warnet/gpu_black_noaug_300_model.pt", map_location=device))
    model.to(device)

    # Dataset & dataloader
    def coco_transform(image, target):
        return transforms.ToTensor()(image), target

    test_dataset = CocoDetection(
        root="C:/Users/azxx/AppData/Local/label-studio/label-studio/media/upload/10",
        annFile="data/test/test_coco_result.json",
        transforms=coco_transform
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: tuple(zip(*x))
    )

    # Evaluasi
    metrics = evaluate_coco(
        model,
        test_dataloader,
        device,
        "data/test/test_coco_result.json",
        "data/test/300test_predictions.json"
    )

    print("\n📊 Hasil Evaluasi COCO (Test set)")
    print(f"mAP @[IoU=0.50:0.95]: {metrics[0]:.4f}")
    print(f"mAP @[IoU=0.50]: {metrics[1]:.4f}")
    print(f"mAP @[IoU=0.75]: {metrics[2]:.4f}")
