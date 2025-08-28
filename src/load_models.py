from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from src.architectures import CardClassifier, CRNN
import torch

from torchvision import models
import streamlit as st

# === Cropping models yolo ===
@st.cache_resource
def load_crop_model():
    model_path = hf_hub_download(
        repo_id="Nicias/card_cropper",  # ton repo Hugging Face
        filename="card_cropper_v1.pt"           # nom exact du fichier dans le repo
    )
    crop_model = YOLO(model_path)
    return crop_model

# === Classification models CNN ===
@st.cache_resource
def load_class_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = hf_hub_download(
        repo_id="Nicias/card_classifier",
        filename="card_classifier_v1.pth"
    )

    model = CardClassifier(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# === Deskew model ResNet18 ===
@st.cache_resource
def load_deskew_model(angles=None, device=None):
    """
    Charge le modèle de deskew avec le bon nombre de classes selon la liste d'angles.
    """
    if angles is None:
        # Par défaut, 25 classes de 0 à 360° par pas de 15°
        angles = [i for i in range(0, 361, 15)]
    if device is None:
        device = torch.device("cpu")
    model_path = hf_hub_download(
        repo_id="Nicias/card_deskewer",
        filename="card_deskewer_v2.pth"
    )
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(angles))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model

# === Text detection model Yolo OBB ===
@st.cache_resource
def load_bio_text_model():
    model_path = hf_hub_download(
        repo_id="Nicias/rec_bio_text_boxes",
        filename="rec_text_v2.pt"
    )
    model = YOLO(model_path)
    return model

# === OCR model CRNN ===
CRNN_ALPHABET = " !'-./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_adeiopsué"
CRNN_IMG_H = 32
CRNN_NC = 1
CRNN_NH = 256
CRNN_NCLASS = len(CRNN_ALPHABET) + 1

@st.cache_resource
def load_ocr_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = hf_hub_download(
        repo_id="Nicias/rec_text",
        filename="rec_text_crnn_v2.pth"
    )
    model = CRNN(CRNN_IMG_H, CRNN_NC, CRNN_NCLASS, CRNN_NH)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model