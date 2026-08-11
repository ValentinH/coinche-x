# Prototype jetable — reconnaissance de cartes

Projet en pause. État et résultats :
[jalon de reconnaissance](../../docs/card-recognition-milestone.md).

Question : le contrat de reconnaissance, le banc et la page de progression
suffisent-ils pour comparer honnêtement des moteurs navigateur ?

```sh
cd prototypes/card-recognition
pnpm install
pnpm dev
```

- `/` : une image → ensemble canonique + diagnostics.
- `/evaluate` : import manuel du manifeste v2 et des images, puis métriques.
- Vérité terrain disponible uniquement dans le module d’évaluation chargé en
  développement. Le build de production désactive cette route.
- Chaque image est vérifiée par SHA-256 avant inférence.
- Le moteur initial est un placeholder déterministe. Il ne prédit aucune carte
  et l’annonce dans ses diagnostics : aucune précision simulée.

Le manifeste attendu est le schéma v2 : `images[].cards` contient directement
les identifiants canoniques. Le schéma v1 mutualisé via `cardSets` est rejeté.

```sh
pnpm check
```
