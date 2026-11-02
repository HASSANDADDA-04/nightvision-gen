# Journal d'Expériences et Résultats Comparatifs — NightVision Gen

> [!IMPORTANT]
> Conformément aux principes de rigueur scientifique et aux consignes du projet, **aucun résultat expérimental fictif n'est inventé**. Les tableaux ci-dessous sont préparés et structurés pour accueillir les mesures réelles dès l'exécution des entraînements GPU sur Google Colab.

---

## 1. Environnement Matériel et Hyperparamètres

### Configuration de la Session d'Entraînement
- **Plateforme d'exécution** : Google Colab (GPU NVIDIA T4 / V100 / A100)
- **Framework** : PyTorch 2.x, Torchvision
- **Précision** : Mixed Precision FP16 (`torch.cuda.amp`)
- **Graine aléatoire (Seed)** : 42 (déterministe)
- **Taille des patchs d'entraînement** : 128 × 128 pixels
- **Nombre de pas de diffusion (T)** : 1 000
- **Calendrier de variance** : Cosine (Nichol & Dhariwal)
- **Optimiseur** : AdamW ($\beta_1=0.9, \beta_2=0.999$, weight_decay = $10^{-4}$)
- **Taux d'apprentissage initial** : $10^{-4}$ avec décroissance cosinusoïdale vers $10^{-6}$

---

## 2. Tableau Comparatif des Performances Visuelles sur LOL-v2

Évaluation effectuée sur les **100 paires réelles de test** du sous-ensemble `Real_captured` de LOL-v2 (jamais vues durant l'entraînement ou la validation).

| Méthode | Conditionnement | Entraîné depuis zéro | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | Temps / img (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Sombre Original** | Aucun | Non | — | — | — | — |
| **Correction Gamma (γ=2.2)** | Aucun | Non | — | — | — | — |
| **CLAHE** | Aucun | Non | — | — | — | — |
| **Modèle A (DDPM)** | Sombre (6 canaux) | Oui | — | — | — | — |
| **Modèle B (+ Retinex)** | Sombre + Retinex (7 canaux) | Oui | — | — | — | — |
| **Modèle C (+ Contours)** | Sombre + Retinex + Sobel (7 canaux) | Oui | — | — | — | — |

*(À compléter après exécution du notebook `05_evaluate_quality.ipynb`)*

---

## 3. Tableau Comparatif de Détection d'Objets sur ExDark

Évaluation effectuée sur le sous-ensemble stratifié d'environ 1 000 images d'ExDark (12 classes) avec le détecteur préentraîné sur COCO (**Faster R-CNN ResNet-50 FPN**) aux poids gelés.

| Image d'Entrée du Détecteur | mAP@0.5 ↑ | Rappel@0.5 ↑ | Précision@0.5 ↑ | Objets Vrais Positifs |
| :--- | :---: | :---: | :---: | :---: |
| **Image Sombre Brute** | — | — | — | — |
| **Image Corrigée Gamma** | — | — | — | — |
| **Image Rehaussée CLAHE** | — | — | — | — |
| **Image Restaurée Modèle A** | — | — | — | — |
| **Image Restaurée Modèle B** | — | — | — | — |
| **Image Restaurée Modèle C** | — | — | — | — |

*(À compléter après exécution du notebook `06_evaluate_detection_exdark.ipynb`)*

---

## 4. Étude d'Ablation sur le Poids de la Perte de Contours $\lambda_{\text{edge}}$ (Modèle C)

Comparaison de l'influence de la contrainte de gradient Sobel sur la qualité de reconstruction et la détection d'objets :

| Valeur de $\lambda_{\text{edge}}$ | PSNR (dB) | SSIM | LPIPS | mAP@0.5 ExDark | Observations qualitatives |
| :---: | :---: | :---: | :---: | :---: | :--- |
| $\lambda = 0.00$ *(équiv. Modèle B)* | — | — | — | — | — |
| $\lambda = 0.01$ | — | — | — | — | — |
| $\lambda = 0.05$ *(valeur standard)* | — | — | — | — | — |
| $\lambda = 0.10$ | — | — | — | — | — |

---

## 5. Mesure de Latence d'Inférence DDIM

Mesuré sur une image 128 × 128 pixels (moyenne sur 50 exécutions après 5 passages d'échauffement) :

| Étapes DDIM ($S$) | Paramètre $\eta$ | Temps GPU (s) | Débit (FPS) | Temps CPU (s) |
| :---: | :---: | :---: | :---: | :---: |
| 25 | 0.0 (déterministe) | — | — | — |
| 50 | 0.0 (déterministe) | — | — | — |
| 100 | 0.0 (déterministe) | — | — | — |

*(À compléter après exécution du script `scripts/benchmark_timing.py` ou notebook 05)*

