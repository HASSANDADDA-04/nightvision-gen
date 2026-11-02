# Description des Jeux de Données — NightVision Gen

Le projet repose sur une séparation stricte des rôles entre deux jeux de données complémentaires : **LOL-v2** et **ExDark**.

---

## 1. LOL-v2 (Low-Light Dataset Version 2)

### 1.1 Origine et Rôle
LOL-v2 fournit des paires d'images réelles capturées dans des scènes à luminosité contrôlée en faisant varier le temps d'exposition et la sensibilité ISO :
- Une image sous-exposée (temps de pose très court).
- Une image de référence bien exposée correspondante (temps de pose long ou flash d'appoint).

Dans NightVision Gen, le sous-ensemble **`Real_captured`** de LOL-v2 est exclusivement utilisé pour :
1. L'apprentissage du générateur DDPM conditionnel.
2. L'évaluation quantitative avec référence (PSNR, SSIM, LPIPS).

### 1.2 Partitions Strictes
Pour garantir l'intégrité de l'évaluation scientifique :
- **Ensemble de Test** : Les **100 paires de test réelles** sont figées et sanctuarisées. Aucun patch n'en est extrait pour l'entraînement, et aucun choix d'hyperparamètres ne s'y appuie.
- **Ensemble d'Entraînement & Validation** : Les couples restants de `Real_captured/Train` sont découpés aléatoirement en 85% pour l'entraînement et 15% pour la validation (surveillance du surapprentissage et choix du meilleur checkpoint).
- **Extraction de Patchs** : Le découpage en patchs 128 × 128 n'intervient qu'après scission des ensembles, garantissant qu'aucun morceau d'une image de test n'apparaisse dans le jeu d'entraînement.

---

## 2. ExDark (Exclusively Dark Dataset)

### 2.1 Origine et Rôle
ExDark (Loh & Chan 2019) est une collection de **7 363 images** prises dans des conditions de très faible éclairage naturel en extérieur et en milieu urbain (nuits sombres, éclairage public lointain, pénombre).

Chaque image est annotée avec des boîtes englobantes (*bounding boxes*) couvrant 12 classes d'objets.

> [!IMPORTANT]
> ExDark n'est **jamais utilisé pour entraîner le DDPM** dans le protocole principal. Il sert uniquement à évaluer l'impact applicatif de la restauration sur un détecteur d'objets en conditions réelles.

### 2.2 Sous-ensemble Stratifié
Afin de concilier représentativité statistique et temps de calcul, un sous-ensemble stratifié de **1 000 images** est sélectionné de façon reproductible (graine fixe = 42), en équilibrant les 12 classes d'objets et les types d'éclairage ambiant.

### 2.3 Correspondance Explicite des Classes (ExDark <-> COCO)
Le détecteur préentraîné sur MS-COCO prend en charge 80 catégories. La table ci-dessous formalise la projection des 12 classes ExDark :

| Classe ExDark | Classe COCO Équivalente | ID Catégorie COCO | Remarques méthodologiques |
| :--- | :--- | :---: | :--- |
| **Bicycle** | `bicycle` | 2 | Correspondance exacte |
| **Boat** | `boat` | 9 | Correspondance exacte |
| **Bottle** | `bottle` | 44 | Correspondance exacte |
| **Bus** | `bus` | 6 | Correspondance exacte |
| **Car** | `car` | 3 | Correspondance exacte |
| **Cat** | `cat` | 16 | Correspondance exacte |
| **Chair** | `chair` | 62 | Correspondance exacte |
| **Cup** | `cup` | 47 | Correspondance exacte |
| **Dog** | `dog` | 17 | Correspondance exacte |
| **Motorbike** | `motorcycle` | 4 | Correspondance exacte |
| **People** | `person` | 1 | Regroupe silhouettes piétonnes |
| **Table** | `dining table` | 67 | Proche, mais exclut les tables de bureau |

---

## 3. Décalage de Domaine (Domain Shift)

Le passage de LOL-v2 à ExDark introduit un décalage de distribution significatif :
- **LOL-v2** : scènes d'intérieur statiques, objets domestiques, bruit de capteur lié aux courtes expositions, paires alignées.
- **ExDark** : scènes d'extérieur dynamiques, véhicules, piétons, lumières ponctuelles (phares, lampadaires), bruit environnemental varié, absence d'image de référence bien éclairée.

Ce décalage de domaine est au cœur de l'analyse expérimentale : il teste la capacité de généralisation réelle du modèle génératif.

