import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
from src import pipeline_module


st.title("Extraction d'informations de cartes biométriques")

choix_mode = option_menu(
    menu_title=None,
    menu_icon="card",
    options=["OCR sur carte", "OCR sur liste de présence"],
    orientation="horizontal"
)

if choix_mode == "OCR sur carte":
    st.subheader("Extraction sur une seule carte")

    uploaded_files = st.file_uploader("Choisissez une ou plusieurs images",type=["jpg", "jpeg", "png"],
                                    accept_multiple_files=True)

    if uploaded_files:
        st.write(f"{len(uploaded_files)} image(s) chargée(s)")

        if st.button("Extraire les informations (batch)"):
            pil_images = [Image.open(f).convert("RGB") for f in uploaded_files]
            progress = st.progress(0, text="Extraction en cours...")
            excel_buffer, total, success, failed = pipeline_module.process_images_to_excel_pil(pil_images)
            progress.progress(100, text="Extraction terminée")

            st.info(f"Total images analysées : {total}")
            st.success(f"Images avec informations extraites : {success}")
            st.warning(f"Images sans informations extraites : {failed}")

            if excel_buffer:
                st.download_button(
                    label="📥 Télécharger le fichier Excel des résultats",
                    data=excel_buffer,
                    file_name="extraction_resultats.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("Aucune information n'a pu être extraite des images chargées.")

elif choix_mode == "OCR sur liste de présence":
    st.subheader("Extraction sur une liste de présence")
    st.info("Cette fonctionnalité n'est pas encore implémentée.")