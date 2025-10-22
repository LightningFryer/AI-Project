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

    print(f"✅ Predictions saved in {output_dir}/")

train_dir = r"dataset\datasets\balraj98\deepglobe-road-extraction-dataset\versions\2\train"
valid_dir = r"dataset\datasets\balraj98\deepglobe-road-extraction-dataset\versions\2\valid"

train_ds = RoadDatasetTrain(train_dir)
valid_ds = RoadDatasetValid(valid_dir)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_ds, batch_size=2, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet()
model.load_state_dict(torch.load("./road_unet_epoch5.pth"))
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

title = "🛰️ Road Accessibility Detector"
description = """
Upload a post-disaster satellite image to detect accessible roads.<br>
The model outputs a binary mask and a visual overlay showing roads in green.
"""

demo = gr.Interface(
    fn=predict_road_mask,
    inputs=gr.Image(label="Upload Satellite Image", type="numpy"),
    outputs=[
        gr.Image(label="Predicted Road Mask", type="numpy"),
        gr.Image(label="Overlay on Satellite Image", type="numpy")
    ],
    title=title,
    description=description,
    examples=None,
)

demo.launch(debug=False, share=True)