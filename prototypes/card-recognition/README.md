# PROTOTYPE — reconnaissance des cartes

## Question

Une approche existante et permissive peut-elle fournir une base hors ligne assez
précise pour éviter l'entraînement d'un modèle spécialisé ?

Le corpus `test-data/card-recognition/smoke` sert ici au développement exploratoire.
Les résultats ne constituent pas une validation indépendante.

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
| Tesseract.js brut | 0/20 exactes ; précision rang 69,8 % ; rappel rang 57,1 % | 345 ms moyenne, 701 ms max sur Mac | rejet |
| PaddleJS OCR | 0/4 rang sur la première photo | 13,2 s chargement ; 1,5 s photo | arrêt anticipé |
| KNN fvannee + OpenCV.js | 5/20 exactes ; précision carte 92,1 % ; rappel carte 34,8 % | 407 ms moyenne, 490 ms max sur Mac | meilleur candidat, précision insuffisante |

Le KNN obtient 5/10 photos exactes sur le jeu `J/Q/K` (précision 94,0 %,
rappel 64,9 %) et 0/10 sur le jeu `V/D/R` (précision 72,7 %, rappel 4,8 %).
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
