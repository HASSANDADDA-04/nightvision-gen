"""Reusable UI components for the Streamlit web application."""

from __future__ import annotations

import streamlit as st


def render_header() -> None:
    """Render application title and descriptive banner."""
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 1.5rem; border: 1px solid #312e81;">
            <h1 style="color: #f8fafc; margin: 0; font-size: 2.2rem; font-weight: 700;">🌙 NightVision Gen</h1>
            <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 1.05rem;">
                Amélioration de scènes nocturnes par <strong>Diffusion Conditionnelle (DDPM / DDIM)</strong> guidée par <strong>Retinex</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_info() -> None:
    """Render scientific summary in sidebar."""
    st.sidebar.markdown("### 🔬 Architecture & Modèles")
    st.sidebar.info(
        """
        - **Modèle A** : DDPM conditionné par l'image sombre
        - **Modèle B** : DDPM + Carte d'illumination Retinex
        - **Modèle C** : Modèle B + Perte de conservation des contours
        - **Échantillonneur** : DDIM accéléré (25 à 100 étapes)
        - **Résolution** : 128 × 128 par patchs avec fenêtrage Hann
        """
    )
