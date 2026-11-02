# Architecture Logicielle et Système — NightVision Gen

Ce document décrit l'organisation modulaire, les interactions entre sous-systèmes et les pipelines de traitement de NightVision Gen.

---

## 1. Vue d'Ensemble de l'Architecture

Le système s'articule autour de 6 modules principaux :
1. **`data/`** : Gestion des partitions, extraction de patchs 128x128 et DataLoaders.
2. **`illumination/`** : Estimation et lissage des cartes Retinex max-RGB.
3. **`models/`** : U-Net conditionnel, embeddings sinusoïdaux et formulation DDPM.
4. **`losses/`** : Perte de prédiction de bruit et contrainte de contours Sobel.
5. **`training/`** : Trainer indépendant, gestion des checkpoints d'état et journalisation.
6. **`sampling/` & `evaluation/`** : Échantillonnage DDIM rapide, métriques de qualité d'image et évaluation de détection gelée COCO.

```mermaid
graph TD
    subgraph Données
        RawData["LOL-v2 / ExDark"] --> SplitMgr["SplitManager (Train / Val / Test)"]
        SplitMgr --> PatchExt["PatchExtractor (128x128)"]
        SplitMgr --> Retinex["RetinexEstimator (Max-RGB + Gauss)"]
    end

    subgraph Modèle
        PatchExt --> Conditioning["ConditioningManager (A: 6ch, B/C: 7ch)"]
        Retinex --> Conditioning
        Conditioning --> UNet["Conditional U-Net (128x128)"]
        DiffProcess["GaussianDiffusion (Cosine schedule)"] --> UNet
    end

    subgraph Entraînement
        UNet --> LossCalc["CombinedLoss (L_diff + lambda_edge * L_edge)"]
        LossCalc --> Trainer["Trainer (AdamW, AMP, EMA)"]
        Trainer --> Checkpoints["CheckpointManager (Drive / Local)"]
    end

    subgraph Inférence & Évaluation
        Checkpoints --> DDIM["DDIMSampler (25, 50, 100 pas)"]
        DDIM --> Reconstruct["Reconstruction pleine image (Hann blending)"]
        Reconstruct --> MetricsEval["Évaluation Qualité (PSNR, SSIM, LPIPS)"]
        Reconstruct --> DetectionEval["Détection COCO gelée (mAP, Rappel)"]
    end
```

---

## 2. Pipeline de Traitement des Données

Pour éviter toute fuite de données (*data leakage*), le découpage entre Train, Validation et Test est appliqué **avant** toute extraction de patchs.

```mermaid
flowchart LR
    A["Images Réelles LOL-v2 (Couples Sombre / Nette)"] --> B{"Partitionnement Strict"}
    B -->|"100 Couples"| C["Ensemble de Test Réel (Réservé pour évaluation)"]
    B -->|"Reste des Couples"| D["Ensemble d'Apprentissage"]
    D -->|"15%"| E["Ensemble de Validation"]
    D -->|"85%"| F["Ensemble d'Entraînement"]
    F --> G["Extraction de Patchs 128x128 (Chevauchement configuré)"]
    E --> H["Extraction de Patchs 128x128 Validation"]
```

---

## 3. Pipeline d'Entraînement du DDPM (Modèles A, B et C)

```mermaid
sequenceDiagram
    autonumber
    participant D as DataLoader (Patchs)
    participant R as Retinex Estimator
    participant G as Forward Diffusion
    participant U as Conditional U-Net
    participant L as Combined Loss
    participant T as Trainer & Optimizer

    D->>G: Image nette x0, Image sombre Is
    opt Si Modèle B ou C
        D->>R: Image sombre Is
        R-->>U: Carte d'illumination L (1 canal)
    end
    G->>G: Tirage aléatoire t ~ U(0, T-1) et bruit epsilon ~ N(0, I)
    G->>U: Tenseur bruité x_t
    U->>U: Concaténation conditionnelle (6ch pour A, 7ch pour B/C)
    U->>L: Prédiction du bruit epsilon_theta
    opt Si Modèle C
        L->>L: Reconstruction x0_hat et calcul perte de contours Sobel
    end
    L->>T: Perte scalaire totale L_total
    T->>U: Rétropropagation + Pas d'optimisation (AdamW + Scaler AMP)
```

---

## 4. Pipeline d'Inférence et de Restauration Pleine Image

Pour toute image de résolution supérieure à 128 × 128, le module d'inférence segmente l'image en tuiles chevauchantes et applique une pondération 2D de Hann pour éliminer tout artéfact de raccord.

```mermaid
flowchart TD
    ImgIn["Image Sombre d'Entrée (Résolution arbitraire H x W)"] --> Decomp["Décomposition en tuiles 128x128 (stride=96)"]
    Decomp --> BatchPatches["Batch de patchs sombres"]
    
    opt Si Modèle B ou C
        ImgIn --> Illum["Calcul Carte Retinex globale"]
        Illum --> DecompIllum["Décomposition tuiles Retinex"]
        DecompIllum --> BatchPatches
    end
    
    BatchPatches --> DDIMStep["Échantillonnage DDIM (S=25, 50 ou 100 pas)"]
    DDIMStep --> RestoredPatches["Patchs restaurés 128x128"]
    RestoredPatches --> Stitching["Reconstruction 2D avec Fenêtrage de Hann"]
    Stitching --> FinalImage["Image Restaurée Finale (H x W, sans coutures)"]
```

