# Rapport projet SMART

## Participants
- Tanguy GIMENEZ
- Raffael DI PIETRO
- Franck GUTMANN

## Expérimentations conduites
Nous avons testé différentes variantes du modèle YOLO d'Ultralytics, notamment YOLO-N, YOLO-S et 
YOLO-M, afin d'évaluer leurs performances respectives sur notre tâche.

### Tests avec YOLO-M
Lors de nos tests avec le modèle YOLO-M, 
nous avons réduit la taille du batch de 16 à 8 afin d'éviter un crash de la carte graphique.
Cependant, nous n'avons pas augmenté le nombre d'epochs, ce qui a limité l'amélioration des performances
du modèle. Cela suggère que nous aurions dû augmenter le nombre d'epochs pour compenser cette réduction
de batch size.

### Comparaison environnements de training
Nous avons expérimenté l'entraînement sur deux environnements matériels différents :

- Apple M3 Pro (MPS)
- NVIDIA RTX 3060 Ti (CUDA)
- Ryzen 7 7730 PRO (CPU)

L'entraînement sur CUDA était significativement plus rapide, réduisant le temps d'une époque de 2 minutes (MPS) à 15 secondes (CUDA), ce qui a grandement facilité l'optimisation et l'affinage du modèle.
Le CPU était quasiment aussi lent que le MPS.

## Pistes d'améliorations

### Optimisation des hyperparamètres
- Augmenter le nombre d'epochs jusqu'à atteindre la stagnation de la loss function.
- Ajuster le nombre d'epochs en fonction du batch size pour compenser.

### Utiliser un modèle plus performant
- Tester YOLO-L ou YOLO-X si les ressources matérielles le permettent, afin de voir si des architectures
    plus complexes apportent une meilleure précision.

### Optimisation matérielle
- Tester l'entraînement sur une machine avec une carte graphique plus puissante (RTX 4090, A100) pour permettre l'utilisation de batch sizes plus élevés.
- Ne pas commencer à travailler sur le projet une semaine avant la date de rendu.

## Lien vers la meilleure version du modèle 
[Meilleure version du modèle](https://app.picsellia.com/0192f6db-86b6-784c-80e6-163debb242d5/model/01936427-df02-721e-89c1-36706efc4ed6/version/0194eb36-0f82-7658-80de-333c9b1894c0)