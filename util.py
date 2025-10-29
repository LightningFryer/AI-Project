import os, cv2, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import gradio as gr
import shutil
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

def evaluate_model_segmentation(model, dataloader, device=None, threshold=0.5):
    """
    Evaluate a PyTorch model on a binary segmentation task (e.g., flood/wildfire masks),
    ensuring no device mismatch errors.

    Args:
        model: trained PyTorch model
        dataloader: PyTorch DataLoader yielding (images, masks)
        device: 'cuda' or 'cpu'; if None, automatically selects
        threshold: probability threshold for converting sigmoid outputs to binary masks
    """
    # Automatically select device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            # Ensure both inputs and masks are on the same device as the model
            images = images.to(device, dtype=torch.float)

            # Some dataloaders (like RoadDatasetValid) return (image, filename)
            # instead of (image, mask). Handle both cases: if `masks` is a
            # tensor, move it to device; otherwise try to load mask files from
            # the dataset `root_dir` using the filenames in `masks`.
            if isinstance(masks, torch.Tensor):
                masks = masks.to(device, dtype=torch.float)
            else:
                # masks is likely a list/tuple of filenames (or a single name)
                names = masks
                # normalize to iterable of strings
                if isinstance(names, (str, bytes)):
                    names = [names]
                # dataloader.dataset usually has root_dir attribute
                root = getattr(dataloader.dataset, 'root_dir', None)
                loaded = []
                for n in names:
                    try:
                        name_str = n.decode() if isinstance(n, bytes) else str(n)
                    except Exception:
                        name_str = str(n)

                    mask_path = None
                    if root is not None:
                        mask_path = os.path.join(root, name_str.replace("_sat.jpg", "_mask.png"))

                    if mask_path and os.path.exists(mask_path):
                        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        if m is None:
                            # fallback to zeros if file unreadable
                            m = np.zeros((512, 512), dtype=np.uint8)
                        else:
                            m = cv2.resize(m, (512, 512))
                        m = (m > 128).astype(np.float32)
                    else:
                        # if we can't find a mask file, use a zero mask and warn
                        m = np.zeros((512, 512), dtype=np.float32)
                    loaded.append(torch.tensor(m).unsqueeze(0))

                # stack into shape (B,1,H,W) and move to device
                masks = torch.stack(loaded, dim=0).to(device, dtype=torch.float)

            outputs = model(images)
            # model may already return probabilities (sigmoid applied). If
            # outputs are in [0,1], treat them as probabilities; otherwise
            # apply sigmoid.
            try:
                if torch.all(outputs >= 0) and torch.all(outputs <= 1):
                    probs = outputs
                else:
                    probs = torch.sigmoid(outputs)
            except Exception:
                probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()  # Binarize predictions

            # Flatten tensors for metric calculation
            y_true.append(masks.view(-1).cpu())
            y_pred.append(preds.view(-1).cpu())

    # Concatenate all batches
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    # Compute metrics
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "IoU": jaccard_score(y_true, y_pred, zero_division=0)
    }

    # Print metrics
    print("Validation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


class RoadDatasetTrain(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = sorted([f for f in os.listdir(root_dir) if f.endswith("_sat.jpg")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        mask_name = img_name.replace("_sat.jpg", "_mask.png")

        img_path = os.path.join(self.root_dir, img_name)
        mask_path = os.path.join(self.root_dir, mask_name)

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize to manageable size
        image = cv2.resize(image, (512, 512))
        mask = cv2.resize(mask, (512, 512))

        # Binarize mask
        mask = (mask > 128).astype(np.float32)

        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        mask = torch.tensor(mask).unsqueeze(0)
        return image, mask
    
def iou_score(pred, target, threshold=0.5):
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

class RoadDatasetValid(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = sorted([f for f in os.listdir(root_dir) if f.endswith("_sat.jpg")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = self.files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
        return image, img_name

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.d1 = DoubleConv(3, 64)
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.u3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.u2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.u1 = DoubleConv(128, 64)
        self.final = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        c1 = self.d1(x)
        c2 = self.d2(self.pool(c1))
        c3 = self.d3(self.pool(c2))
        c4 = self.d4(self.pool(c3))

        u3 = self.up3(c4)
        u3 = torch.cat([u3, c3], dim=1)
        u3 = self.u3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, c2], dim=1)
        u2 = self.u2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, c1], dim=1)
        u1 = self.u1(u1)

        return torch.sigmoid(self.final(u1))

def predict_and_save(model, valid_loader, device, output_dir="./predictions"):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for imgs, names in tqdm(valid_loader, desc="Generating predictions"):
            imgs = imgs.to(device)
            preds = model(imgs)
            preds = (preds > 0.5).float().cpu().numpy()

            for i in range(len(names)):
                mask = (preds[i][0] * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_dir, names[i].replace("_sat.jpg", "_pred.png")), mask)

    print(f"Predictions saved in {output_dir}/")

train_dir = r"dataset\datasets\balraj98\deepglobe-road-extraction-dataset\versions\2\train"
valid_dir = r"dataset\datasets\balraj98\deepglobe-road-extraction-dataset\versions\2\valid"

train_ds = RoadDatasetTrain(train_dir)
valid_ds = RoadDatasetValid(valid_dir)
new_ds = RoadDatasetTrain(r"train_test_dir")

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_ds, batch_size=2, shuffle=False)
new_loader = DataLoader(new_ds, batch_size=2, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet()
model.load_state_dict(torch.load("./road_unet_epoch15.pth"))
model.to(device)
# Train only on training set
# train_model(model, train_loader, device, epochs=5, lr=1e-4)

# Generate binary masks for validation images
# predict_and_save(model, valid_loader, device)

def predict_road_mask(image):
    # Resize + preprocess
    img_resized = cv2.resize(image, (512, 512))
    inp = torch.tensor(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    inp = inp.to(device)

    # Predict mask
    with torch.no_grad():
        pred = model(inp)
    mask = (pred > 0.5).float().cpu().numpy()[0, 0]

    # Make overlay (roads in green)
    mask_rgb = np.zeros_like(img_resized)
    mask_rgb[..., 1] = (mask * 255).astype(np.uint8)  # green channel
    overlay = cv2.addWeighted(img_resized, 0.8, mask_rgb, 0.6, 0)

    return (mask * 255).astype(np.uint8), overlay

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

def plot_segmentation_metrics(model, dataloader, device=None, save_dir="./metrics_plots"):
    """
    Generates and saves ROC and Precision–Recall curves for the segmentation model.
    Uses flattened pixel-level predictions and targets.
    """
    os.makedirs(save_dir, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    y_true = []
    y_score = []

    print("Collecting predictions for ROC/PR computation...")
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Metrics Eval"):
            images = images.to(device, dtype=torch.float)

            # Handle valid_loader returning filenames
            if not isinstance(masks, torch.Tensor):
                names = masks
                if isinstance(names, (str, bytes)):
                    names = [names]
                root = getattr(dataloader.dataset, 'root_dir', None)
                loaded = []
                for n in names:
                    mask_path = os.path.join(root, n.replace("_sat.jpg", "_mask.png"))
                    if os.path.exists(mask_path):
                        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        m = cv2.resize(m, (512, 512))
                        m = (m > 128).astype(np.float32)
                    else:
                        m = np.zeros((512, 512), dtype=np.float32)
                    loaded.append(torch.tensor(m).unsqueeze(0))
                masks = torch.stack(loaded, dim=0)

            masks = masks.to(device, dtype=torch.float)
            outputs = model(images)
            probs = torch.sigmoid(outputs)

            y_true.append(masks.view(-1).cpu().numpy())
            y_score.append(probs.view(-1).cpu().numpy())

    y_true = np.concatenate(y_true)
    y_score = np.concatenate(y_score)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Road Segmentation")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "roc_curve.png"), dpi=300)
    plt.close()

    # Precision–Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_prec = average_precision_score(y_true, y_score)

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR Curve (AP = {avg_prec:.4f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve for Road Segmentation")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "precision_recall_curve.png"), dpi=300)
    plt.close()

    print(f"✅ ROC and PR curves saved in {save_dir}/")
    print(f"AUC: {roc_auc:.4f}, Average Precision: {avg_prec:.4f}")

    return {"AUC": roc_auc, "Average Precision": avg_prec}


def copy_train_subset(train_root, dest_root, fraction=0.2, seed=42,
                      sat_suffix="_sat.jpg", mask_suffix="_mask.png",
                      overwrite=False, dry_run=False):
    """
    Sample a fraction of satellite images from `train_root` and copy both the
    satellite images and their corresponding masks to `dest_root`.

    Args:
        train_root: path to the train folder containing *_sat.jpg and *_mask.png
        dest_root: destination directory to create and copy files into
        fraction: fraction of satellite files to sample (0< fraction <=1)
        seed: RNG seed for reproducible sampling
        sat_suffix: suffix used by satellite files (default "_sat.jpg")
        mask_suffix: suffix used by mask files (default "_mask.png")
        overwrite: if True, overwrite files in destination
        dry_run: if True, do not copy, only report what would be done

    Returns:
        dict with keys: sampled_count, copied_sat_count, copied_mask_count,
        missing_masks_count, sampled_list, copied_sat, copied_mask, missing_masks
    """
    os.makedirs(dest_root, exist_ok=True)

    all_sat = sorted([f for f in os.listdir(train_root) if f.endswith(sat_suffix)])
    if not all_sat:
        print(f"No satellite files found in {train_root} with suffix {sat_suffix}")
        return {
            "sampled_count": 0,
            "copied_sat_count": 0,
            "copied_mask_count": 0,
            "missing_masks_count": 0,
            "sampled_list": [],
            "copied_sat": [],
            "copied_mask": [],
            "missing_masks": [],
        }

    import random
    random.seed(seed)
    k = max(1, int(len(all_sat) * float(fraction)))
    sampled = sorted(random.sample(all_sat, k))

    copied_sat = []
    copied_mask = []
    missing_masks = []

    for sat in sampled:
        src_sat = os.path.join(train_root, sat)
        dst_sat = os.path.join(dest_root, sat)

        mask_name = sat.replace(sat_suffix, mask_suffix)
        src_mask = os.path.join(train_root, mask_name)
        dst_mask = os.path.join(dest_root, mask_name)

        # Dry-run: only check existence and report
        if dry_run:
            if os.path.exists(src_sat):
                copied_sat.append(sat)
            if os.path.exists(src_mask):
                copied_mask.append(mask_name)
            else:
                missing_masks.append(mask_name)
            continue

        # Copy sat image
        try:
            if os.path.exists(src_sat):
                if not os.path.exists(dst_sat) or overwrite:
                    shutil.copy2(src_sat, dst_sat)
                copied_sat.append(sat)
            else:
                # If the source satellite file is missing, skip
                print(f"Warning: source sat missing: {src_sat}")
                continue
        except Exception as e:
            print(f"Failed to copy sat {src_sat} -> {dst_sat}: {e}")
            continue

        # Copy mask if exists
        if os.path.exists(src_mask):
            try:
                if not os.path.exists(dst_mask) or overwrite:
                    shutil.copy2(src_mask, dst_mask)
                copied_mask.append(mask_name)
            except Exception as e:
                print(f"Failed to copy mask {src_mask} -> {dst_mask}: {e}")
                missing_masks.append((mask_name, str(e)))
        else:
            missing_masks.append(mask_name)

    result = {
        "sampled_count": len(sampled),
        "copied_sat_count": len(copied_sat),
        "copied_mask_count": len(copied_mask),
        "missing_masks_count": len(missing_masks),
        "sampled_list": sampled,
        "copied_sat": copied_sat,
        "copied_mask": copied_mask,
        "missing_masks": missing_masks,
    }

    print(f"Sampled {result['sampled_count']} sat images from {train_root}: copied_sat={result['copied_sat_count']}, copied_mask={result['copied_mask_count']}, missing_masks={result['missing_masks_count']}")
    if dry_run:
        print("Dry-run mode: no files were actually copied.")

    return result


if __name__ == "__main__":
    try:
        from multiprocessing import freeze_support
        freeze_support()
    except Exception:
        pass

    # Run metrics generation (safe to execute as the main module)
    # metrics = plot_segmentation_metrics(model, new_loader, device)
    # print(metrics)

    

    title = "Road Accessibility Detector"
    description = """
    Upload a post-disaster satellite image to detect accessible roads.<br>
    The model outputs a binary mask and a visual overlay showing roads in green.
    """

    demo = gr.Interface(
        fn=predict_road_mask,
        inputs=gr.Image(label="Upload Satellite Image", type="numpy", height=512, width=512),
        outputs=[
            gr.Image(label="Predicted Road Mask", type="numpy", height=512, width=512),
            gr.Image(label="Overlay on Satellite Image", type="numpy", height=512, width=512)
        ],
        title=title,
        description=description,
        examples=None,
    )

    demo.launch(debug=False, share=False)