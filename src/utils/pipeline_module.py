import io
import pandas as pd
import cv2
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from .load_models import load_crop_model, load_deskew_model, load_class_model, load_bio_text_model, load_ocr_model
import os
from io import BytesIO

# === Fonctions du pipeline ===
def crop_card(image_input, conf_threshold=0.25):
    """
    Crop card from image and return list of cropped images (as numpy arrays).
    Accepts either a file path (str) or a PIL Image.
    """
    model = load_crop_model()
    # Si c'est un chemin, on lit normalement
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        results = model.predict(source=image_input, save=False, conf=conf_threshold)
    # Si c'est une image PIL, on convertit en numpy array et on passe l'image directement
    elif isinstance(image_input, Image.Image):
        image = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        results = model.predict(source=image, save=False, conf=conf_threshold)
    else:
        raise ValueError("image_input must be a file path or PIL Image")
    boxes = results[0].boxes
    cropped_imgs = []
    if boxes is not None and len(boxes) > 0:
        coords = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for i, (box, cls_id) in enumerate(zip(coords, classes)):
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            cropped_imgs.append(crop)
    return cropped_imgs

def deskew_image(img, angles=None, device='cpu', confidence_threshold=0.5):
    """
    Deskew a single cropped image (numpy array) and return the deskewed PIL image, predicted angle, and confidence.
    Always returns a tuple of three values.
    """
    if angles is None:
        angles = [0, 45, 90, 135, 180, 225, 270, 315]
    angle_to_class = {angle: idx for idx, angle in enumerate(angles)}
    class_to_angle = {idx: angle for angle, idx in angle_to_class.items()}
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    model = load_deskew_model(device=device,angles=angles)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, 1).item()
        predicted_angle = class_to_angle[predicted_class]
        confidence = probabilities[0, predicted_class].item()

    deskew_angle = predicted_angle
    deskewed_img = pil_img.rotate(-deskew_angle, expand=True)
    gray = deskewed_img.convert("L")
    bbox = gray.getbbox()
    if bbox:
        deskewed_img = deskewed_img.crop(bbox)
    if confidence >= confidence_threshold:
        return deskewed_img, predicted_angle, confidence
    else:
        return None, predicted_angle, confidence

def classify_card(img, device='cpu', image_size=(50, 50)):
    """
    Predict the card class for a cropped image (numpy array) and return the predicted class.
    """
    class_labels = ['CARTE BIOMETRIQUE', 'CARTE CIP', 'CNI', 'PASSEPORT']
    # Convert numpy image to RGB if needed
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img, image_size)
    img_tensor = torch.from_numpy(img_resize).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    model = load_class_model(device=device)

    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        outputs = model(img_tensor)
        predicted_class = torch.argmax(outputs, 1)
        return class_labels[predicted_class.item()]

def rec_carte_bio_text_boxes(pil_img, conf_threshold=0.25):
    """
    Detect and crop text boxes from deskewed PIL image, return dict of class->PIL image.
    """
    temp_path = "_temp_text_img.jpg"
    pil_img.save(temp_path)
    model = load_bio_text_model()
    image = cv2.imread(temp_path)
    results = model.predict(source=temp_path, save=False, conf=conf_threshold)
    boxes = results[0].boxes
    text_boxes = {}
    if boxes is not None and len(boxes) > 0:
        coords = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        for i, (box, cls_id) in enumerate(zip(coords, classes)):
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            class_name = model.names[cls_id].lower()
            text_boxes[class_name] = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    os.remove(temp_path)
    return text_boxes

def ocr_text(text_boxes):
    """
    Extract text from an image using PP-OCRv5 server model.
    """
    fields = {"nom": None, "prenom": None, "date_expiration": None, "numero_carte": None}
    recognizer = load_ocr_model()
    for field in fields.keys():
        box_img = text_boxes.get(field)
        if box_img:
            # Save to buffer for OCR
            buf = BytesIO()
            box_img.save(buf, format='JPEG')
            buf.seek(0)
            temp_path = "_temp_ocr_img.jpg"
            with open(temp_path, "wb") as f:
                f.write(buf.read())
            output = recognizer.predict(input=temp_path)
            if output and len(output) > 0:
                fields[field] = output[0].get('rec_text')
            os.remove(temp_path)
    return fields

# === Pipeline principal ===
def pipeline_predict_texts_pil(pil_image):
    # 1. Crop
    cropped_imgs = crop_card(pil_image)
    if not cropped_imgs:
        return None
    cropped_img = cropped_imgs[0]
    # 2. Deskew
    deskewed_img, angle, conf = deskew_image(cropped_img)
    if deskewed_img is None:
        return None
    # 3. Classify
    card_class = classify_card(cropped_img)
    if card_class != "CARTE BIOMETRIQUE":
        return None
    # 4. Extract text boxes
    text_boxes = rec_carte_bio_text_boxes(deskewed_img)
    if not text_boxes:
        return None
    # 5. Recognize text
    fields = ocr_text(text_boxes)
    return fields

# === Traitement batch pour Streamlit ===
def process_images_to_excel_pil(pil_images):
    results = []
    for pil_image in pil_images:
        infos = pipeline_predict_texts_pil(pil_image)
        if infos is not None and all(x is not None and x != '' for x in infos.values()):
            results.append(infos)
    if results:
        df = pd.DataFrame(results)
        output = io.BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        return output, len(pil_images), len(results), len(pil_images) - len(results)
    else:
        return None, len(pil_images), 0, len(pil_images)