import streamlit as st
from PIL import Image
from src import pipeline_module
from streamlit_option_menu import option_menu

st.title("DATA TOOLBOX")

# Sidebar pour la sélection du mode
image ="LogoAbmsPSI.jpg"
st.logo(image)
st.sidebar.title("Navigation")
st.sidebar.text("Choisir le document à générer :")
ocr_carte_btn = st.sidebar.button("Demande de paiement",width=200)
ocr_liste_btn = st.sidebar.button("Etat de paiement",width=200)

# Gestion de l'état pour la sélection
if 'choix_mode' not in st.session_state:
    st.session_state['choix_mode'] = None
if ocr_carte_btn:
    st.session_state['choix_mode'] = "Demande de paiement"
if ocr_liste_btn:
    st.session_state['choix_mode'] = "Etat de paiement"

choix_mode = st.session_state['choix_mode']

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
                    df, total, success, failed = pipeline_module.process_images_to_excel_pil(pil_images)
                    progress.progress(100, text="Extraction terminée")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.info(f"Total images analysées : {total}")
                    with col2:
                        st.success(f"Images avec informations extraites : {success}")
                    with col3:
                        st.warning(f"Images sans informations extraites : {failed}")

                    if df is not None and not df.empty:
                        st.dataframe(df)
                        output_xlsx = "paiement.xlsx"
                        pipeline_module.gen_demande_paiement(df, nom_activite, demandeur, superviseur, output_xlsx)
                        with open(output_xlsx, "rb") as f:
                            st.download_button(
                                label="📥 Télécharger la demande de paiement",
                                data=f,
                                file_name=output_xlsx,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    else:
                        st.warning("Aucune information n'a pu être extraite des images chargées.")

        elif demande_options == "Utiliser une liste de présence":
            st.subheader("Extraction sur une liste de présence")
            st.info("Cette fonctionnalité n'est pas encore implémentée.")

elif choix_mode == "Etat de paiement":
    st.info("Cette fonctionnalité n'est pas encore implémentée.")