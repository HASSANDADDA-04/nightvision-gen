# Organisation des Données — NightVision Gen

Ce répertoire gère les jeux de données utilisés pour l'entraînement, la validation et l'évaluation du projet.

## 1. Structure attendue

```
data/
├── raw/
│   ├── LOLv2/
│   │   ├── Real_captured/
│   │   │   ├── Train/
│   │   │   │   ├── Low/       # Images sombres (.png)
│   │   │   │   └── Normal/    # Images de référence bien exposées (.png)
│   │   │   └── Test/
│   │   │       ├── Low/       # 100 images sombres réelles de test
│   │   │       └── Normal/    # 100 images de référence réelles de test
│   │   └── Synthetic/         # (Optionnel, non utilisé dans le protocole principal)
│   └── ExDark/
│       ├── Bicycle/
│       ├── Boat/
│       ├── Bottle/
│       ├── Bus/
│       ├── Car/
│       ├── Cat/
│       ├── Chair/
│       ├── Cup/
│       ├── Dog/
│       ├── Motorbike/
│       ├── People/
│       ├── Table/
│       └── Annotations/       # Fichiers .txt d'annotations ExDark
└── processed/
    ├── LOLv2/
    │   ├── train_patches/     # Patchs 128x128 extraits du train LOL-v2
    │   ├── val_patches/       # Patchs 128x128 extraits de la partition validation
    │   ├── test/              # 100 paires de test LOL-v2 (images complètes)
    │   └── illumination_maps/ # Cartes Retinex précalculées
    └── ExDark/
        ├── stratified_subset/ # ~1 000 images sombres sélectionnées
        └── annotations_coco/  # Format COCO JSON pour l'évaluation mAP
```

## 2. Rôles et séparation stricte des jeux de données

| Caractéristique | LOL-v2 (Real_captured) | ExDark |
| :--- | :--- | :--- |
| **Nature** | Paires sombre / référence bien exposée | Images sombres en scènes réelles avec classes d'objets |
| **Usage** | Entraînement DDPM & métriques de restauration (PSNR, SSIM, LPIPS) | Évaluation de détection (mAP, rappel) avec détecteur COCO gelé |
| **Résolution vue** | Patchs 128 × 128 pixels | Pleine image traitée par patchs avec chevauchement |
| **Partition test** | 100 paires réservées (intouchables durant le train) | ~1 000 images stratifiées (12 classes) |

> [!IMPORTANT]
> Ne jamais utiliser ExDark pour entraîner le DDPM dans le protocole principal.
> Ne jamais utiliser le jeu de test de LOL-v2 pour régler les hyperparamètres ou extraire les patchs d'entraînement.

## 3. Téléchargement et préparation automatique

Vous pouvez préparer les données automatiquement via le script :
```bash
python scripts/prepare_data.py --config configs/data.yaml
```
Ou exécuter le notebook `notebooks/01_data_exploration.ipynb` pour visualiser les paires et valider l'intégrité des données.

