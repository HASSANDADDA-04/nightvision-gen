"""Streamlit interactive web demonstration for NightVision Gen.

Features:
    - Low-light image upload (PNG, JPG, BMP)
    - Variant selection (Model A, B, or C)
    - DDIM sampling steps slider (10 to 100)
    - Side-by-side visualization: Dark Input, Retinex Illumination Map, Restored Output
    - Real-time inference latency reporting
    - Download button for enhanced images
    - Graceful error handling (missing checkpoint, invalid format, CPU fallback)
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from app.components.ui import render_header, render_sidebar_info
from nightvision.sampling.inference import InferenceService

st.set_page_config(
    page_title="NightVision Gen — Démonstration",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Chargement du modèle en mémoire...")
def load_service(
    checkpoint_path: str, model_variant: str, ddim_steps: int, device: str
) -> InferenceService:
    return InferenceService(
        checkpoint_path=checkpoint_path,
        model_variant=model_variant,
        device=device,
        ddim_steps=ddim_steps,
        eta=0.0,
    )


def main() -> None:
    render_header()
    render_sidebar_info()

    # --- Sidebar Configuration ---
    st.sidebar.markdown("### ⚙️ Paramètres d'Inférence")
    variant = st.sidebar.selectbox(
        "Modèle à utiliser",
        options=["Model_A", "Model_B", "Model_C"],
        index=2,
        help="Sélectionnez l'une des 3 variantes étudiées.",
    )
    model_letter = variant[-1]

    ddim_steps = st.sidebar.slider(
        "Nombre d'étapes DDIM",
        min_value=10,
        max_value=100,
        value=25,
        step=5,
        help="Moins d'étapes = inférence plus rapide ; plus d'étapes = fidélité accrue.",
    )

    device_pref = st.sidebar.radio(
        "Accélération matérielle",
        options=["auto", "cuda", "cpu"],
        index=0,
        horizontal=True,
    )

    # Checkpoint path
    default_ckpt = Path(f"artifacts/checkpoints/model_{model_letter}/checkpoint_best.pt")
    custom_ckpt = st.sidebar.text_input("Chemin du checkpoint (.pt)", value=str(default_ckpt))

    ckpt_exists = Path(custom_ckpt).is_file()
    if not ckpt_exists:
        st.sidebar.warning(
            f"⚠️ Checkpoint introuvable : `{custom_ckpt}`.\n"
            "Le modèle utilisera une initialisation aléatoire (mode démo/test)."
        )

    # --- Main Content Area ---
    col_upload, col_preview = st.columns([1, 2])

    with col_upload:
        st.markdown("#### 📥 1. Importer une image sombre")
        uploaded_file = st.file_uploader(
            "Choisissez une image nocturne ou sombre",
            type=["png", "jpg", "jpeg", "bmp"],
        )

        sample_btn = st.button("🖼️ Charger une image d'exemple synthétique")

        image_to_process = None
        if uploaded_file is not None:
            try:
                image_to_process = Image.open(uploaded_file).convert("RGB")
            except Exception as e:
                st.error(f"Erreur de lecture du fichier : {e}")
        elif sample_btn:
            # Generate deterministic synthetic low-light gradient
            arr = np.zeros((128, 128, 3), dtype=np.uint8)
            for y in range(128):
                for x in range(128):
                    arr[y, x] = [int(15 + 25 * (x / 128.0)), int(10 + 20 * (y / 128.0)), 20]
            image_to_process = Image.fromarray(arr, mode="RGB")

    with col_preview:
        if image_to_process is not None:
            st.markdown(
                f"#### 🔎 Dimensions de l'image : `{image_to_process.width} × {image_to_process.height}` px"
            )
            if image_to_process.width > 128 or image_to_process.height > 128:
                st.info(
                    "ℹ️ L'image sera découpée en patchs 128×128 avec fenêtrage Hann pour une reconstruction sans coutures."
                )

            run_btn = st.button(
                "🚀 Lancer la Restauration NightVision Gen",
                type="primary",
                use_container_width=True,
            )

            if run_btn:
                try:
                    with st.spinner(
                        f"Restauration avec le Modèle {model_letter} ({ddim_steps} étapes DDIM)..."
                    ):
                        service = load_service(
                            checkpoint_path=custom_ckpt,
                            model_variant=model_letter,
                            ddim_steps=ddim_steps,
                            device=device_pref,
                        )
                        service.set_ddim_steps(ddim_steps)

                        restored_img, illum_img, latency = service.enhance_image(image_to_process)

                    st.success(
                        f"✅ Restauration réussie en **{latency:.3f} s** sur **{service.device.type.upper()}** !"
                    )

                    # Display comparison columns
                    if illum_img is not None:
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown("**Image Sombre (Entrée)**")
                            st.image(image_to_process, use_container_width=True)
                        with c2:
                            st.markdown("**Carte Retinex (Guide)**")
                            st.image(illum_img, use_container_width=True)
                        with c3:
                            st.markdown(f"**Restaurée ({variant})**")
                            st.image(restored_img, use_container_width=True)
                    else:
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Image Sombre (Entrée)**")
                            st.image(image_to_process, use_container_width=True)
                        with c2:
                            st.markdown(f"**Restaurée ({variant})**")
                            st.image(restored_img, use_container_width=True)

                    # Download button
                    buf = io.BytesIO()
                    restored_img.save(buf, format="PNG")
                    st.download_button(
                        label="💾 Télécharger l'image restaurée (PNG)",
                        data=buf.getvalue(),
                        file_name=f"nightvision_{variant}_{ddim_steps}steps.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"Une erreur est survenue lors de la génération : {e}")
                    st.exception(e)
        else:
            st.info(
                "Veuillez importer une image sombre ou cliquer sur 'Charger une image d'exemple' pour commencer."
            )


if __name__ == "__main__":
    main()
