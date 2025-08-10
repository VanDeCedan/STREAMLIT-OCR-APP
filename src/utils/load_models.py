from ultralytics import YOLO
import torch
from paddleocr import TextRecognition
from torch import nn
import torch.nn.functional as F_nn
from torchvision import models

class ResNet18Classification(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

class CardClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(CardClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F_nn.relu(self.conv1(x))     # [B, 32, 50, 50] → [B, 32, 25, 25]
        x = self.pool(x)
        x = F_nn.relu(self.conv2(x))     # [B, 64, 25, 25] → [B, 64, 12, 12]
        x = self.pool(x)
        x = F_nn.relu(self.conv3(x))     # [B, 128, 12, 12] → [B, 128, 6, 6]
        x = self.pool(x)
        x = self.flatten(x)           # [B, 128*6*6]
        x = F_nn.relu(self.fc1(x))
        x = self.fc2(x)  # Match 'softmax' from TensorFlow
        return x

def load_crop_model():
    model_path = "../models/card_cropper.pt"
    crop_model = YOLO(model_path)
    return crop_model

def load_deskew_model(device='cpu',angles=None):
    # Default angles ()
    if angles is None:
        angles = [0,45,90,135,180,225,270,315]
    
    # Create angle to class mapping
    model_path = "../models/card_deskewer.pth"
    
    # Load and initialize the model
    model = ResNet18Classification(num_classes=len(angles))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

def load_class_model(device='cpu'):
    model_path = "../models/card_classifier.pth"

    model = CardClassifier(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def load_bio_text_model():
    model_path = "../models/rec_bio_text.pt"
    model = YOLO(model_path)
    return model

def load_ocr_model():
    model_dir = "../models/PP-OCRv5_server_rec"
    model_name = "PP-OCRv5_server_rec"
    # Initialize the text recognizer with the model directory and name
    recognizer = TextRecognition(model_name=model_name ,model_dir=model_dir)
    return recognizer