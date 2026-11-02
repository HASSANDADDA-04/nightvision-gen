# Méthodologie Scientifique — NightVision Gen

## 1. Contexte et Problématique

La capture d'images en conditions de très faible éclairage (nuit, brouillard, obscurité urbaine) pose des défis majeurs pour les systèmes de vision par ordinateur. Ces images souffrent de :
- Une luminance globale très faible et un rapport signal-sur-bruit (SNR) dégradé.
- Un bruit de capteur et bruit thermique amplifiés.
- Une perte de contraste locale et globale.
- L'effacement ou le masquage des détails fins, silhouettes et contours nécessaires à la détection d'objets (piétons, véhicules, cyclistes).

L'objectif du projet **NightVision Gen** est de concevoir un système génératif de restauration d'images nocturnes fondé sur un **modèle probabiliste de diffusion par débruitage (DDPM)** conditionnel, guidé par une carte d'illumination de type **Retinex**, et doté d'une contrainte de **préservation des contours**.

---

## 2. Fondements Théoriques

### 2.1 Théorie Retinex et Carte d'Illumination
Selon le modèle classique de Retinex (Land & McCann) :
$$I(x, y) = R(x, y) \cdot L(x, y)$$

Où :
- $I(x, y)$ est l'image sombre observée.
- $R(x, y)$ est la réflectance intrinsèque de la scène (propriétés physiques des surfaces, invariante à l'éclairage).
- $L(x, y)$ est la carte d'illumination spatiale.

Dans NightVision Gen, nous adoptons l'estimation max-RGB non supervisée :
$$L_{\text{raw}}(x, y) = \max \left( R(x, y), G(x, y), B(x, y) \right)$$

Pour éliminer les textures haute fréquence tout en préservant les transitions d'éclairage globales, nous appliquons un filtre gaussien spatial séparable de variance $\sigma = 3.0$ et noyau $K = 15$ :
$$L(x, y) = \frac{L_{\text{smooth}}(x, y) - \min(L)}{\max(L) - \min(L)}$$

La carte $L(x, y) \in [0, 1]$ sert de **guide d'illumination conditionnel**, aidant le réseau de diffusion à discerner les régions sous-exposées des zones possédant un éclairage d'appoint.

---

### 2.2 Modèle Génératif de Diffusion DDPM Conditionnel

Le générateur de diffusion est un DDPM (Ho et al. 2020) **entraîné depuis zéro** (sans poids préentraînés).

#### Processus Avant (Forward Diffusion)
Une chaîne de Markov discrète en $T = 1000$ étapes ajoute progressivement du bruit gaussien à l'image bien éclairée de référence $x_0 \sim q(x_0)$ :
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t \mathbf{I})$$

En posant $\alpha_t = 1 - \beta_t$ et $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ :
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

L'échantillon bruité direct s'écrit :
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

Nous utilisons un calendrier de variances cosine (Nichol & Dhariwal) évitant la dégradation brutale des fréquences moyennes observée avec le calendrier linéaire.

#### Processus Inverse Conditionnel (Reverse Diffusion)
Le modèle U-Net conditionnel $\epsilon_\theta(x_t, t, c)$ apprend à prédire le bruit $\epsilon$ injecté, conditionné par les indices $c$ :
$$p_\theta(x_{t-1} | x_t, c) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t, c), \sigma_t^2 \mathbf{I})$$

---

### 2.3 Échantillonnage Accéléré avec DDIM

Pour l'inférence pratique et l'évaluation temporelle, nous exploitons un échantillonneur DDIM (Song et al. 2020) non-markovien. À partir d'une sous-séquence de $S$ pas de temps $\tau_1, \tau_2, \dots, \tau_S$ (avec $S \in \{25, 50, 100\}$) et d'un paramètre stochastique $\eta = 0.0$ (déterministe) :

$$x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}} \hat{x}_0 + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}} - \sigma_{\tau_i}^2} \epsilon_\theta(x_{\tau_i}) + \sigma_{\tau_i} \epsilon$$

Où l'image nette estimée est :
$$\hat{x}_0 = \frac{x_{\tau_i} - \sqrt{1 - \bar{\alpha}_{\tau_i}} \epsilon_\theta(x_{\tau_i})}{\sqrt{\bar{\alpha}_{\tau_i}}}$$

---

## 3. Protocole Comparatif des Trois Variantes (A / B / C)

Pour analyser rigoureusement la contribution scientifique de chaque composant, trois variantes distinctes sont entraînées **indépendamment depuis zéro**, sous des budgets de calcul et protocoles rigoureusement identiques :

| Variante | Conditionnement Entrée U-Net | Canaux d'Entrée | Fonction Objectif ($\mathcal{L}_{\text{total}}$) | Question Scientifique |
| :--- | :--- | :---: | :--- | :--- |
| **Modèle A** | $x_t$ (3) + Image sombre $I_s$ (3) | **6** | $\mathcal{L}_{\text{diff}} = \| \epsilon - \epsilon_\theta \|_1$ | Quel est le niveau de performance de base du DDPM conditionnel ? |
| **Modèle B** | $x_t$ (3) + $I_s$ (3) + Carte Retinex $L$ (1) | **7** | $\mathcal{L}_{\text{diff}} = \| \epsilon - \epsilon_\theta \|_1$ | L'estimation d'illumination Retinex améliore-t-elle la fidélité visuelle et la convergence ? |
| **Modèle C** | $x_t$ (3) + $I_s$ (3) + Carte Retinex $L$ (1) | **7** | $\mathcal{L}_{\text{diff}} + \lambda_{\text{edge}} \mathcal{L}_{\text{edge}}(\hat{x}_0, x_0)$ | La conservation explicite des contours préserve-t-elle mieux les objets pour la détection ? |

### Formulation de la Perte de Contours (Modèle C)
La perte de contours utilise des filtres de Sobel horizontaux et verticaux $G_x$ et $G_y$ pour extraire les gradients de magnitude :
$$|\nabla I| = \sqrt{(I * G_x)^2 + (I * G_y)^2 + \epsilon_0}$$
$$\mathcal{L}_{\text{edge}}(\hat{x}_0, x_0) = \frac{1}{N} \| |\nabla \hat{x}_0| - |\nabla x_0| \|_1$$

Le poids $\lambda_{\text{edge}}$ est paramétrable dans les fichiers YAML (`configs/train_C.yaml`, typiquement $0.05$).

---

## 4. Protocole d'Évaluation Expérimentale

1. **Évaluation Visuelle sur LOL-v2 (100 paires réelles de test réservées)** :
   - PSNR (Peak Signal-to-Noise Ratio en dB)
   - SSIM (Structural Similarity Index Measure)
   - LPIPS (Learned Perceptual Image Patch Similarity)
2. **Évaluation de Détection sur ExDark (~1 000 images stratifiées, 12 classes)** :
   - Détecteur préentraîné **Faster R-CNN ResNet-50 FPN** aux poids strictement gelés.
   - Correspondance des classes ExDark avec COCO.
   - Mesure de **mAP@0.5** et du **Rappel@0.5** avant et après restauration.
3. **Mesure du temps d'inférence** :
   - Synchronisation CUDA précise pour quantifier le temps par image à 25, 50 et 100 pas DDIM.

