# Limites et Risques Méthodologiques — NightVision Gen

Une analyse critique et objective des contraintes intrinsèques au projet est indispensable pour situer les résultats dans leur juste contexte scientifique.

---

## 1. Faible Volume de Données d'Entraînement Appariées
- **Constat** : Le jeu de données LOL-v2 `Real_captured` ne compte que quelques centaines de paires pour l'apprentissage.
- **Impact** : Les modèles de diffusion entraînés depuis zéro (*from scratch*) nécessitent habituellement des dizaines de milliers d'images pour apprendre des distributions visuelles riches.
- **Atténuation mise en œuvre** : L'extraction de multiples patchs 128 × 128 avec chevauchement et des augmentations synchronisées (rotations, réflexions) permettent de multiplier artificiellement la diversité spatiale sans créer de fuite de données entre partitions.

---

## 2. Décalage de Domaine (LOL-v2 vers ExDark)
- **Constat** : Le modèle DDPM apprend les motifs d'illumination sur des scènes d'intérieur (meubles, pièces closes), puis est évalué sur des scènes urbaines extérieures nocturnes (rues, véhicules, piétons).
- **Impact** : L'illumination prédite peut être mal calibrée face à des sources de lumière directe (phares de voiture, néons) qui n'existaient pas dans les scènes d'apprentissage.
- **Atténuation mise en œuvre** : Le conditionnement par Retinex isole la tendance globale de la lumière et rend le générateur plus robuste aux variations d'ambiance.

---

## 3. Limites des Métriques d'Évaluation

### 3.1 PSNR et SSIM (Métriques avec référence)
- Le PSNR et le SSIM pénalisent lourdement le moindre décalage spatial de quelques pixels ou une variation infime de tonalité. Une image légèrement lissée peut obtenir un meilleur PSNR qu'une image aux contours nets contenant un léger bruit haute fréquence.

### 3.2 NIQE (Métrique sans référence)
- Le score NIQE mesure la conformité statistique d'une image par rapport à des modèles gaussiens généralisés (MSCN).
- **Point de vigilance** : Une image excessivement floue ou sur-lissée peut afficher un score NIQE artificiellement bas (favorable). C'est pourquoi **NIQE ne doit jamais être interprété isolément**, mais systématiquement confronté aux performances du détecteur d'objets.

---

## 4. Coût Computationnel et Temps d'Inférence
- Contrairement aux méthodes directes en une seule passe (Gamma, CLAHE, ou réseaux feed-forward type RetinexNet / Zero-DCE), les modèles de diffusion nécessitent un processus itératif inverse.
- Même accéléré avec DDIM à 25 ou 50 pas, le temps de calcul par image reste supérieur à plusieurs centaines de millisecondes sur GPU, rendant le système inadapté à la vidéo temps réel haute cadence (e.g. 60 FPS) sans matériel dédié ou quantification poussée (TensorRT, FP8).

---

## 5. Détecteur d'Objets Gelé et Ambiguïtés de Classes
- Le protocole impose de garder les poids du détecteur Faster R-CNN strictement gelés. Si la restauration amplifie un artefact qui ressemble à une texture parasite, le détecteur peut générer un faux positif.
- De plus, certaines catégories ExDark comme `People` regroupent à la fois des piétons debout, des personnes assises ou des silhouettes lointaines, là où COCO distingue `person` avec des critères spécifiques.

