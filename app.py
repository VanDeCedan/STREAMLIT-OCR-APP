# 🔧 Forcer le répertoire de config Ultralytics sur Streamlit Cloud
import os
from pathlib import Path

CFG_DIR = "/tmp/Ultralytics"
os.environ["YOLO_CONFIG_DIR"] = CFG_DIR
Path(CFG_DIR).mkdir(parents=True, exist_ok=True)

import streamlit as st
from PIL import Image
from src.pipeline_module import process_images_to_text_pil
from streamlit_option_menu import option_menu
from src.gen_demande_momo import gen_momo_fr_card
from src.gen_etat_pymt import create_etat_paiement_xlsx

st.title("DATA TOOLBOX")

# Sidebar pour la sélection du mode
image ="LogoAbmsPSI.jpg"
st.logo(image)
Acceuil_btn = st.sidebar.button("Accueil",icon="🏠")
st.sidebar.text("Choisir le document à générer :")
demand_paiement_btn = st.sidebar.button("Demande de paiement",width=200)
etat_paiement_btn = st.sidebar.button("Etat de paiement",width=200)

# Gestion de l'état pour la sélection
if 'choix_mode' not in st.session_state:
    st.session_state['choix_mode'] = None
if demand_paiement_btn:
    st.session_state['choix_mode'] = "Demande de paiement"
if etat_paiement_btn:
    st.session_state['choix_mode'] = "Etat de paiement"
if Acceuil_btn:
    st.session_state['choix_mode'] = None
    

choix_mode = st.session_state['choix_mode']

if st.session_state['choix_mode'] == None:
    st.write("# Bienvenu sur le data toolbox! 👋")


    st.markdown(
        """
        Ceci est un outil de deep learning pour créer des documents 
        administratif.
        **👈 Selectionner ce que vous voulez faire à gauche** 
        ### Attention !
        - Ceci est une version beta, il peut y avoir des bugs.
        - Les modèles de deep learning peuvent faire des erreurs.
        - Vérifier toujours les documents générés.
        ### Important
        - Utiliser des images de bonne qualité (pas floues, bien éclairées).
        - Les pièces d'identité supportées sont : Carte d'identité nationale (CNI), Carte d'identité biométrique, Carte CIP, Passeport.
        - N'utiliser que des documents dont vous avez **l'autorisation des propriétaires**. Pour en savoir plus Contactez le DPDP de l'ABMS Maxiès AHODEKON, 01 67 52 53 49, mahodekon@abmsbj.org
    """
    )

if choix_mode == "Demande de paiement":
    demande_options = option_menu("MENU", ["Utiliser des cartes", "Utiliser une liste de présence"],
                                icons=["card-text", "list-task"], menu_icon="cast", default_index=0, orientation="horizontal")

    if demande_options == "Utiliser des cartes":
        st.subheader("Extraction depuis des cartes")
        # Champ pour le titre de l'activité
        col1, col2, col3 = st.columns(3)
        with col1:
            nom_activite = st.text_input("Titre de l'activité (motif de paiement)")
        with col2:
            demandeur = st.text_input("Nom du demandeur")
        with col3:
            superviseur = st.text_input("Nom du superviseur")

        uploaded_files = st.file_uploader("Choisissez une ou plusieurs images de pièce d'identité",
                                        type=["jpg", "jpeg", "png"],
                                        accept_multiple_files=True)

        if uploaded_files:
            st.write(f"{len(uploaded_files)} image(s) chargée(s)")

            if st.button("Extraire les informations"):
                if not nom_activite.strip():
                    st.error("Veuillez d'abord renseigner le titre de l'activité (motif de paiement) avant d'extraire les informations.")
                elif not demandeur.strip():
                    st.error("Veuillez renseigner le nom du demandeur.")
                elif not superviseur.strip():
                    st.error("Veuillez renseigner le nom du superviseur.")
                else:
                    pil_images = [Image.open(f).convert("RGB") for f in uploaded_files]
                    progress = st.progress(0, text="Extraction en cours...")
                    df, total, success, failed = process_images_to_text_pil(pil_images)
                    progress.progress(100, text="Extraction terminée")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"Total images analysées : {total}")
                    with col2:
                        st.success(f"Traitée avec succès : {success}")
                    with col3:
                        st.warning(f"Erronés : {failed}")

                    if df is not None and not df.empty:
                        # Modifier les noms de colonnes de df avec la premiere lettre en majuscule et les underscores en espaces
                        df1 = df.copy() # faire une copie de df
                        df1.columns = [col.replace('_', ' ').capitalize() for col in df.columns]
                        st.dataframe(df1)
                        output_xlsx = "paiement.xlsx"
                        gen_momo_fr_card(df, nom_activite, demandeur, superviseur, output_xlsx)

                        with open(output_xlsx, "rb") as f:
                            def demand_payment_callback():
                                @st.dialog("Success")
                                def success_dialog():
                                    st.success("Le téléchargement a été réalisé !", icon="✅")
                                st.session_state['choix_mode'] = None
                                success_dialog()
                            st.download_button(
                                label="📥 Télécharger la demande de paiement",
                                data=f,
                                file_name=output_xlsx,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                on_click=demand_payment_callback
                            )

                    else:
                        st.warning("Aucune information n'a pu être extraite des images chargées.")

        elif demande_options == "Utiliser une liste de présence":
            st.subheader("Extraction sur une liste de présence")
            st.info("Cette fonctionnalité n'est pas encore implémentée.")

elif choix_mode == "Etat de paiement":
    st.subheader("Génération de l'état de paiement")
    col1, col2 = st.columns(2)
    with col1:
        titre_activite = st.text_input("Titre de l'activité")
        lieu_activite = st.text_input("Lieu de l'activité")
    with col2:
        date_activite = st.text_input("Date de l'activité (ex: 01/09/2025)")
        date_paiement = st.text_input("Date de paiement (ex: 02/09/2025)")

    # Initialiser la liste des fichiers dans la session si besoin
    if 'etat_paiement_files' not in st.session_state:
        st.session_state['etat_paiement_files'] = []

    uploaded_files = st.file_uploader("Choisissez une ou plusieurs images de pièce d'identité",
                                      type=["jpg", "jpeg", "png"],
                                      accept_multiple_files=True)


    # Synchronisation suppression : si l'utilisateur retire une image de l'uploader, on la retire aussi de la session
    if uploaded_files is not None:
        # On ne garde que les fichiers présents dans uploaded_files
        uploaded_names = {f.name for f in uploaded_files}
        st.session_state['etat_paiement_files'] = [f for f in st.session_state['etat_paiement_files'] if f.name in uploaded_names]
        # Ajout des nouveaux fichiers (évite doublons)
        existing_names = {f.name for f in st.session_state['etat_paiement_files']}
        new_files = [f for f in uploaded_files if f.name not in existing_names]
        st.session_state['etat_paiement_files'].extend(new_files)

    total_files = len(st.session_state['etat_paiement_files'])
    st.write(f"{total_files} image(s) en attente de traitement")

    if total_files > 0:
        if st.button("Extraire les informations (état de paiement)"):
            if not titre_activite.strip():
                st.error("Veuillez renseigner le titre de l'activité.")
            elif not lieu_activite.strip():
                st.error("Veuillez renseigner le lieu de l'activité.")
            elif not date_activite.strip():
                st.error("Veuillez renseigner la date de l'activité.")
            elif not date_paiement.strip():
                st.error("Veuillez renseigner la date de paiement.")
            else:
                pil_images = [Image.open(f).convert("RGB") for f in st.session_state['etat_paiement_files']]
                progress = st.progress(0, text="Extraction en cours...")
                df, total, success, failed = process_images_to_text_pil(pil_images)
                progress.progress(100, text="Extraction terminée")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"Total images analysées : {total}")
                with col2:
                    st.success(f"Traitée avec succès : {success}")
                with col3:
                    st.warning(f"Erronés : {failed}")

                if df is not None and not df.empty:
                    df1 = df.copy()
                    df1.columns = [col.replace('_', ' ').capitalize() for col in df.columns]
                    st.dataframe(df1)
                    output_xlsx = "etat_paiement.xlsx"
                    create_etat_paiement_xlsx(df, output_xlsx, titre_activite, date_activite, lieu_activite, date_paiement)
                    with open(output_xlsx, "rb") as f:
                        def etat_paiement_callback():
                            @st.dialog("Success")
                            def success_dialog():
                                st.success("Le téléchargement a été réalisé !", icon="✅")
                            st.session_state['choix_mode'] = None
                            st.session_state['etat_paiement_files'] = []  # Réinitialise la liste après téléchargement
                            success_dialog()
                        st.download_button(
                            label="📥 Télécharger l'état de paiement",
                            data=f,
                            file_name=output_xlsx,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            on_click=etat_paiement_callback
                        )
                else:
                    st.warning("Aucune information n'a pu être extraite des images chargées.")