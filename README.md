# 🌙 NightVision Gen

**Amélioration de scènes nocturnes par diffusion conditionnelle (DDPM / DDIM) guidée par Retinex.**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB.svg?logo=python)](https://python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: ruff](https://img.shields.io/badge/linter-ruff-brightgreen.svg)](https://github.com/astral-sh/ruff)

---

## 1. Présentation et Objectifs Scientifiques

**NightVision Gen** est une application modulaire de restauration d'images nocturnes et à faible luminosité combinant la théorie **Retinex** et les **modèles probabilistes de diffusion par débruitage (DDPM / DDIM)**.

Le système vise un double objectif :
1. **Qualité Visuelle** : Restauration de la luminosité, du contraste et réduction du bruit sur le jeu de données apparié **LOL-v2** (mesuré par PSNR, SSIM et LPIPS).
2. **Utilité Applicative** : Préservation des objets critiques (piétons, voitures, vélos) pour la détection d'objets sur le jeu de données réel nocturne **ExDark**, évalué avec un détecteur **COCO gelé** (Faster R-CNN).

### Trois Variantes Étudiées (Entraînées Indépendamment Depuis Zéro) :
- **Modèle A** : DDPM conditionné par l'image sombre seule (concaténation 6 canaux).
- **Modèle B** : DDPM conditionné par l'image sombre + carte d'illumination Retinex $L(x,y) = \max(R,G,B)$ lissée (concaténation 7 canaux).
- **Modèle C** : Modèle B + pénalité explicite de préservation des contours de Sobel ($\mathcal{L}_{\text{diff}} + \lambda_{\text{edge}} \mathcal{L}_{\text{edge}}$).

---

## 2. Structure du Répertoire

```text
nightvision-gen/
├── configs/                  # Fichiers de configuration YAML
│   ├── data.yaml             # Paramètres des jeux de données et patchs
│   ├── model.yaml            # Hyperparamètres U-Net et diffusion
│   ├── train_A.yaml          # Config entraînement Modèle A (Base)
│   ├── train_B.yaml          # Config entraînement Modèle B (+ Retinex)
│   ├── train_C.yaml          # Config entraînement Modèle C (+ Contours)
│   └── evaluation.yaml       # Config métriques et détection COCO
├── data/                     # Jeux de données (non versionnés dans Git)
│   ├── raw/LOLv2/            # Paires réelles LOL-v2 (Train / Test)
│   ├── raw/ExDark/           # Images nocturnes annotées ExDark
│   └── processed/            # Patchs 128x128 et manifestes JSON
├── notebooks/                # Notebooks Jupyter interactifs documentés
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_retinex.ipynb
│   ├── 03_baselines_gamma_clahe.ipynb
│   ├── 04_train_ddpm_colab.ipynb        # Notebook principal d'entraînement Colab
│   ├── 05_evaluate_quality.ipynb
│   └── 06_evaluate_detection_exdark.ipynb
├── src/nightvision/          # Package Python modulaire
│   ├── data/                 # Datasets, patchs et découpages stricts
│   ├── illumination/         # Estimation Retinex max-RGB
│   ├── models/               # U-Net conditionnel, DDPM, embeddings
│   ├── losses/               # Perte diffusion L1/L2 et perte de contours Sobel
│   ├── training/             # Trainer, checkpoints et journalisation
│   ├── sampling/             # Échantillonneur accéléré DDIM et inférence
│   ├── evaluation/           # Métriques (PSNR, SSIM, LPIPS, NIQE) et détection
│   └── utils/                # Config, reproductibilité et visualisation
├── scripts/                  # Scripts CLI exécutables
│   ├── prepare_data.py
│   ├── generate_illumination.py
│   ├── train.py
│   ├── resume_training.py
│   ├── generate_images.py
│   ├── evaluate_quality.py
│   ├── evaluate_detection.py
│   └── export_results.py
├── app/                      # Application interactive Streamlit
│   ├── streamlit_app.py
│   ├── inference_service.py
│   └── components/
├── tests/                    # Tests unitaires automatisés PyTest (100% CPU)
├── docs/                     # Documentation académique et scientifique
│   ├── methodology.md
│   ├── architecture.md
│   ├── experiments.md
│   ├── dataset_description.md
│   └── limitations.md
├── requirements.txt
├── requirements-colab.txt
└── pyproject.toml
```

---

## 3. Installation Locale

### Prérequis
- Python 3.9, 3.10 ou 3.11
- Git

### Installation
```bash
# 1. Cloner le dépôt
git clone https://github.com/HASSANDADDA-04/nightvision-gen.git
cd nightvision-gen

# 2. Créer et activer l'environnement virtuel
python -m venv .venv
# Sur Windows :
.\.venv\Scripts\activate
# Sur Linux / macOS :
source .venv/bin/activate

# 3. Installer les dépendances et le package en mode éditable
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

---

## 4. Lancement des Tests Unitaires

Tous les tests unitaires s'exécutent sur **CPU** avec des données synthétiques légères, sans nécessiter de GPU ni le téléchargement préalable des datasets :

```bash
pytest tests/ -v
```

---

## 5. Exécution des Entraînements sur Google Colab (GPU)

Le développement du code source est effectué en local, tandis que les entraînements intensifs s'exécutent sur Google Colab.

1. Ouvrez le notebook : [`notebooks/04_train_ddpm_colab.ipynb`](notebooks/04_train_ddpm_colab.ipynb).
2. Activez l'accélérateur GPU : **Exécution > Modifier le type d'exécution > T4 GPU**.
3. Exécutez les cellules séquentiellement :
   - Montage de Google Drive : un seul dossier `MyDrive/nightvision-gen` (datasets LOL-v2 et ExDark, checkpoints, logs et images générées).
   - Clonage automatique du dépôt GitHub.
   - Sélection du modèle à entraîner (**A**, **B** ou **C**) via le formulaire interactif.
   - Entraînement depuis zéro avec sauvegarde automatique des checkpoints (`checkpoint_best.pt`, `checkpoint_latest.pt`) sur Google Drive.
   - En cas d'interruption ou d'expiration de session Colab, activez simplement `RESUME_TRAINING = True` pour reprendre l'entraînement au pas exact où il s'était arrêté.

---

## 6. Utilisation des Scripts en Ligne de Commande

### Entraînement local ou serveur :
```bash
# Entraîner le Modèle A (Base DDPM)
python scripts/train.py --config configs/train_A.yaml

# Entraîner le Modèle B (DDPM + Retinex)
python scripts/train.py --config configs/train_B.yaml

# Entraîner le Modèle C (DDPM + Retinex + Contours Sobel)
python scripts/train.py --config configs/train_C.yaml

# Smoke test rapide sur CPU avec tenseurs synthétiques
python scripts/train.py --config configs/train_A.yaml --synthetic --epochs 2
```

### Restauration d'une image avec DDIM :
```bash
python scripts/generate_images.py \
    --image path/to/dark_image.png \
    --model_variant C \
    --checkpoint artifacts/checkpoints/model_C/checkpoint_best.pt \
    --ddim_steps 50 \
    --output artifacts/generated_images/restored.png
```

---

## 7. Application Web Interactive Streamlit

L'interface Streamlit permet de visualiser et de comparer interactivement les restaurations des modèles A, B et C :

```bash
streamlit run app/streamlit_app.py
```

Fonctionnalités :
- Importation d'une image sombre personnelle ou chargement d'un exemple.
- Sélection du modèle (A, B ou C) et réglage du nombre d'étapes DDIM (10 à 100).
- Affichage simultané de l'image sombre, du guide Retinex et de la sortie restaurée.
- Reconstruction automatique par patchs 128×128 avec fenêtrage de Hann pour les images haute résolution.
- Téléchargement direct de l'image restaurée.

