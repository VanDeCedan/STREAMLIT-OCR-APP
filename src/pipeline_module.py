import pandas as pd
import src.modules as modules

# === Pipeline principal ===
def pipeline_predict_texts_pil(pil_image):
    # 1. Crop
    cropped_imgs = modules.crop_card(pil_image)
    if not cropped_imgs:
        return None
    cropped_img = cropped_imgs[0]
    # 2. Deskew
    deskewed_img, angle, conf = modules.deskew_image(cropped_img)
    if deskewed_img is None:
        return None
    # 3. Classify
    card_class = modules.classify_card(cropped_img)
    # 4. Extract text boxes
    text_boxes = modules.rec_text(deskewed_img)
    if not text_boxes:
        return None
    # 5. Recognize text
    fields = modules.ocr_text(text_boxes)
    # Ajoute la classe prédite au dictionnaire des champs
    fields["card_class"] = card_class
    return fields

# === Traitement batch pour Streamlit ===
def process_images_to_text_pil(pil_images):
    results = []
    for pil_image in pil_images:
        infos = pipeline_predict_texts_pil(pil_image)
        if infos is not None and all(x is not None and x != '' for k, x in infos.items() if k != "card_class"):
            results.append(infos)
    if results:
        df = pd.DataFrame(results)

        # Formater les dates d'expiration de type 'DDMMYYYY' en 'DD/MM/YYYY'
        if 'date_expiration' in df.columns:
            def format_date_exp(val):
                if isinstance(val, str) and len(val) == 8 and val.isdigit():
                    return val[:2] + '/' + val[2:4] + '/' + val[4:]
                return val
            df['date_expiration'] = df['date_expiration'].apply(format_date_exp)

        # Nettoyage colonne numero_carte : supprimer espaces
        if "numero_carte" in df.columns:
            df["numero_carte"] = df["numero_carte"].astype(str).str.replace(" ", "", regex=False)
        # Supprimer lignes avec numero_carte < 8 caractères
        if "numero_carte" in df.columns:
            df = df[df["numero_carte"].str.len() >= 8]
        # Supprimer lignes avec nom < 3 caractères
        if "nom" in df.columns:
            df = df[df["nom"].astype(str).str.len() >= 3]
        # Supprimer lignes avec date_expiration non valide
        if "date_expiration" in df.columns:
            pattern = r"^(\d{2}[ /]\d{2}[ /]\d{4})$"
            df = df[df["date_expiration"].astype(str).str.match(pattern)]

        return df, len(pil_images), len(df), len(pil_images) - len(df)
    else:
        return None, len(pil_images), 0, len(pil_images)