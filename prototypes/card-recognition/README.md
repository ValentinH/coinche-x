# PROTOTYPE — reconnaissance des cartes

## Question

Une approche existante et permissive peut-elle fournir une base hors ligne assez
précise pour éviter l'entraînement d'un modèle spécialisé ?

Le corpus `test-data/card-recognition/smoke` sert ici au développement exploratoire.
Les résultats ne constituent pas une validation indépendante.

```sh
pnpm --dir prototypes/card-recognition validate:manifest
```

Cette validation impose une liste `cards` propre à chaque image.

## Baseline Tesseract.js

```sh
pnpm --dir prototypes/card-recognition benchmark:ocr
```

## Candidat PaddleJS OCR

```sh
pnpm --dir prototypes/card-recognition benchmark:paddle:serve
```

Puis ouvrir
`http://127.0.0.1:4174/prototypes/card-recognition/?file=xiaomi-pad-6_jqk_16.jpg`.
Ajouter `all=1` pour lancer les 20 photos.

## Baseline KNN + OpenCV.js

Le script récupère les 354 gabarits MIT de
`fvannee/android-cards-image-recognition`, puis les compacte en 221 612 octets.

```sh
pnpm --dir prototypes/card-recognition benchmark:knn:serve
```

Puis ouvrir
`http://127.0.0.1:4175/prototypes/card-recognition/knn.html?all=1`.

## Résultats exploratoires

| Candidat | Résultat smoke | Latence observée | Verdict |
| --- | --- | --- | --- |
| Tesseract.js brut | 0/20 exactes ; précision rang 72,0 % ; rappel rang 58,9 % | 350 ms moyenne, 714 ms max sur Mac | rejet |
| PaddleJS OCR | 0/4 rang sur la première photo | 13,2 s chargement ; 1,5 s photo | arrêt anticipé |
| KNN fvannee + OpenCV.js | 6/20 exactes ; précision carte 99,2 % ; rappel carte 37,5 % | 427 ms moyenne, 721 ms max sur Mac | meilleur candidat, rappel insuffisant |

Le KNN obtient 6/10 photos exactes sur le jeu `J/Q/K` (précision 100 %,
rappel 69,0 %) et 0/10 sur le jeu `V/D/R` (précision 90,9 %, rappel 6,0 %).
Trois gabarits `V/D/R` CC0, puis trois gabarits calibrés sur une photo smoke,
n'améliorent pas matériellement ce second jeu.

Le port Chrome charge 15 712 618 octets d'OpenCV.js et 221 612 octets de
gabarits. Sur Brave desktop : 1,11 s à froid et 0,38 s à chaud pour une photo de
quatre cartes. Avec un ralentissement CPU ×4 : 4,12 s à froid, 1,30 s à chaud ;
une photo de 32 cartes prend 2,23 s à 2 000 px, ou 1,90 s de calcul à 1 600 px.
Ce ralentissement est une approximation, pas une mesure sur Xiaomi Pad 6.

Conclusion : aucune approche existante testée ne couvre les deux jeux. Le
prochain essai pertinent est un petit détecteur de coins entraîné avec des
données en ligne permissives, sans demander de nouvelles photos.
