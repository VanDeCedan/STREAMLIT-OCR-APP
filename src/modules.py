import src.load_models as lm
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

# === Cropping ===
def crop_card(image_input, conf_threshold=0.25):
    """
    Crop card from image and return list of cropped images (as numpy arrays).
    Accepts either a file path (str) or a PIL Image.
    """
    model = lm.load_crop_model()
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

# === Classification ===
def classify_card(img, image_size=(50, 50)):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lm.load_class_model(device=device)

    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        outputs = model(img_tensor)
        predicted_class = torch.argmax(outputs, 1)
        return class_labels[predicted_class.item()]

# === Deskew ===
def deskew_image(img, confidence_threshold=0.5):
    """
    Deskew a single cropped image (numpy array) and return the deskewed PIL image, predicted angle, and confidence.
    Always returns a tuple of three values.
    """
    """
    Deskew une image (numpy array) avec le modèle de classification d'angle.
    Retourne (deskewed_image, predicted_angle, confidence)
    """
    # Par défaut, 25 classes de 0 à 360° par pas de 15°
    angles = [i for i in range(0, 361, 15)]
    angle_to_class = {angle: idx for idx, angle in enumerate(angles)}
    class_to_angle = {idx: angle for angle, idx in angle_to_class.items()}
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    model = lm.load_deskew_model(angles=angles, device=device)

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

# === Text detection ===
def rec_text(pil_img, conf_threshold=0.25):
    """
    Detect and crop text boxes from deskewed PIL image, return dict of class->PIL image.
    """
    # Convert PIL image to numpy RGB
    image = np.array(pil_img.convert("RGB"))
    model = lm.load_bio_text_model()
    results = model.predict(source=image, save=False, conf=conf_threshold)
    r = results[0]
    text_boxes = {}
    # OBB oriented boxes
    if hasattr(r, 'obb') and r.obb is not None:
        obb = r.obb
        if hasattr(obb, 'xyxyxyxy') and obb.xyxyxyxy is not None and len(obb.xyxyxyxy) > 0:
            boxes = obb.xyxyxyxy.cpu().numpy() if hasattr(obb.xyxyxyxy, 'cpu') else obb.xyxyxyxy
            classes = obb.cls.cpu().numpy().astype(int) if hasattr(obb.cls, 'cpu') else obb.cls.astype(int)
            for poly, cls_id in zip(boxes, classes):
                pts = np.array(poly, np.int32).reshape((-1, 2))
                rect = cv2.boundingRect(pts)
                x, y, w, h = rect
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                crop_img = cv2.bitwise_and(image, image, mask=mask)[y:y+h, x:x+w]
                # Optionally, crop tighter using the polygon
                class_name = model.names[cls_id].lower() if hasattr(model, 'names') else str(cls_id)
                text_boxes[class_name] = Image.fromarray(crop_img)
            return text_boxes
        # Fallback axis-aligned
        elif hasattr(obb, 'xyxy') and obb.xyxy is not None and len(obb.xyxy) > 0:
            boxes = obb.xyxy.cpu().numpy() if hasattr(obb.xyxy, 'cpu') else obb.xyxy
            classes = obb.cls.cpu().numpy().astype(int) if hasattr(obb.cls, 'cpu') else obb.cls.astype(int)
            for box, cls_id in zip(boxes, classes):
                x1, y1, x2, y2 = map(int, box)
                crop = image[y1:y2, x1:x2]
                class_name = model.names[cls_id].lower() if hasattr(model, 'names') else str(cls_id)
                text_boxes[class_name] = Image.fromarray(crop)
            return text_boxes
    # Fallback to classic .boxes if no OBB
    if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, 'cpu') else r.boxes.xyxy
        classes = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes.cls, 'cpu') else r.boxes.cls.astype(int)
        for box, cls_id in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            class_name = model.names[cls_id].lower() if hasattr(model, 'names') else str(cls_id)
            text_boxes[class_name] = Image.fromarray(crop)
    return text_boxes

# === OCR ===
CRNN_ALPHABET = " !'-./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_adeiopsué"

def ocr_crnn_predict(model, image, alphabet=CRNN_ALPHABET, device=None):
    if device is None:
        device = torch.device("cpu")
    if isinstance(image, Image.Image):
        image = image.convert('L').resize((128,32))
        image = np.array(image, dtype=np.float32) / 255.0
        image = (image - 0.5) / 0.5
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            image = image.mean(axis=2)
        image = Image.fromarray(image.astype(np.uint8)).convert('L').resize((128,32))
        image = np.array(image, dtype=np.float32) / 255.0
        image = (image - 0.5) / 0.5
    else:
        raise ValueError("image must be PIL.Image or numpy array")
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0).to(device)
    model.eval()
    idx_to_char = {idx+1: char for idx, char in enumerate(alphabet)}
    with torch.no_grad():
        output = model(image)
        probs = output.softmax(2)
        out = output.permute(1, 0, 2)
        pred_idx = out.softmax(2).cpu().detach().numpy().argmax(axis=2)[0]
        prev = -1
        decoded = []
        confs = []
        for t, idx in enumerate(pred_idx):
            if idx != prev and idx != 0:
                decoded.append(idx)
                confs.append(probs[t,0,idx].item())
            prev = idx
        pred_str = ''.join([idx_to_char[i] for i in decoded if i in idx_to_char])
        confidence = float(np.mean(confs)) if confs else 0.0
    return pred_str, confidence

def ocr_text(text_boxes):
    fields = {"nom": None, "prenom": None, "date_expiration": None, "numero_carte": None}
    model = lm.load_ocr_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for field in fields.keys():
        box_img = text_boxes.get(field)
        if box_img is not None:
            pred, conf = ocr_crnn_predict(model, box_img, device=device)
            fields[field] = pred if pred else None
    return fields