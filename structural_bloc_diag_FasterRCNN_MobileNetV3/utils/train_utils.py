import torch
import csv
import os
from tqdm.auto import tqdm


def save_metrics_csv(file_path, epoch, metrics):
    fieldnames = ["epoch"] + list(metrics.keys())
    file_exists = os.path.exists(file_path)

    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({"epoch": epoch, **metrics})


def train_one_epoch(model, train_dl, optimizer, epoch_num, n_total_epoch, device, metrics_path, lambda_offdiag):
    with tqdm(train_dl, unit="batch", miniters=300) as tepoch:
        epoch_loss = 0
        _classifier_loss = 0
        _loss_box_reg = 0
        _loss_rpn_box_reg = 0
        _loss_objectness = 0

        for data in tepoch:
            tepoch.set_description(f"Train:Epoch {epoch_num}/{n_total_epoch}")
            imgs = []
            targets = []

            for img, ann in zip(*data):
                imgs.append(img.to(device))
                processed = {}
                boxes = []
                labels = []
                n_empty_boxes = 0
                for obj in ann:
                    x, y, w, h = obj["bbox"]
                    x1, y1, x2, y2 = x, y, x + w, y + h # FasterRCNN's convention is xmin, ymin, xmax, ymax
                    boxes.append([x1, y1, x2, y2])
                    labels.append(obj["category_id"])

                if len(boxes) == 0:
                    imgs.pop()
                    n_empty_boxes += 1
                    #print(n_empty_boxes)
                    continue

                processed["boxes"] = torch.tensor(boxes, dtype=torch.float32).to(device)
                processed["labels"] = torch.tensor(labels, dtype=torch.int64).to(device)
                targets.append(processed)

            loss_dict = model(imgs, targets)
            loss = sum(v for v in loss_dict.values())

            if lambda_offdiag is not None:
                loss += lambda_offdiag * model.roi_heads.box_head.get_off_diag_loss()

            classifier_loss = loss_dict.get("loss_classifier", torch.tensor(0)).item()
            loss_box_reg = loss_dict.get("loss_box_reg", torch.tensor(0)).item()
            loss_objectness = loss_dict.get("loss_objectness", torch.tensor(0)).item()
            loss_rpn_box_reg = loss_dict.get("loss_rpn_box_reg", torch.tensor(0)).item()

            epoch_loss += loss.item()
            _classifier_loss += classifier_loss
            _loss_box_reg += loss_box_reg
            _loss_objectness += loss_objectness
            _loss_rpn_box_reg += loss_rpn_box_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(
                total_loss=epoch_loss,
                loss_classifier=_classifier_loss,
                boxreg_loss=_loss_box_reg,
                obj_loss=_loss_objectness,
                rpn_boxreg_loss=_loss_rpn_box_reg,
            )


        metrics = {
            "total_loss": epoch_loss,
            "classifier_loss": _classifier_loss,
            "box_reg_loss": _loss_box_reg,
            "objectness_loss": _loss_objectness,
            "rpn_box_reg_loss": _loss_rpn_box_reg
        }
        save_metrics_csv(metrics_path, epoch_num, metrics)