import os
import torch
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Create a folder in your project
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # current script directory
MODEL_DIR = os.path.join(BASE_DIR, "pretrained_models")
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Download & save ResNet50
# -----------------------------
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet.fc = torch.nn.Identity()  # remove final layer
resnet_path = os.path.join(MODEL_DIR, "resnet50.pth")
torch.save(resnet.state_dict(), resnet_path)
resnet.to(device).eval()
print(f"ResNet50 weights saved at {resnet_path}")

# -----------------------------
# Download & save EfficientNetB0
# -----------------------------
efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
efficientnet.classifier = torch.nn.Identity()  # remove final layer
efficientnet_path = os.path.join(MODEL_DIR, "efficientnet_b0.pth")
torch.save(efficientnet.state_dict(), efficientnet_path)
efficientnet.to(device).eval()
print(f"EfficientNetB0 weights saved at {efficientnet_path}")
