import torch
import json


def evaluate_model(model, data_loader, device, output_path):
    model.eval()
    coco_dt_list = []

    with torch.no_grad():
        for imgs, targets in data_loader:
            imgs = [img.to(device) for img in imgs]
            outputs = model(imgs)

            for output, target in zip(outputs, targets):
                if not target:
                    continue
                image_id = target[0]["image_id"]
                boxes = output["boxes"].cpu()
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()

                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    coco_dt_list.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO expects [x, y, width, height]
                        "score": float(score),
                    })

    # Save predictions to a JSON file for COCOeval
    with open(output_path, "w") as f:
        json.dump(coco_dt_list, f)