# Coinche

Compagnon de comptage pour une partie de Coinche jouée physiquement.

## Language

**Pli**:
Les quatre cartes jouées durant un tour, une par joueur, et remportées par une équipe.

**Rang**:
L’identité numérique ou figurative d’une carte : 7, 8, 9, 10, Valet, Dame, Roi ou As. Les coins français `V/D/R` et anglais `J/Q/K` représentent les mêmes rangs.
_Avoid_: Valeur

**Dix de der**:
Le bonus de 10 points attribué à l’équipe qui remporte le huitième et dernier pli.
_Avoid_: 10 de der, dernier pli

**Enchère**:
La proposition d’un joueur, composée d’un objectif et d’une couleur ou d’un mode d’atout.
_Avoid_: Annonce

**Contrat**:
L’enchère finale retenue à l’issue des enchères et portée par l’équipe preneuse.
_Avoid_: Enchère finale, annonce

**Équipe preneuse**:
L’équipe qui porte le contrat.
_Avoid_: Attaque

**Équipe défendante**:
L’équipe opposée à l’équipe preneuse.
_Avoid_: Défense

**Responsable du score**:
Le premier joueur de l’équipe photographiée, qui contrôle l’application pendant toute la partie. Ce rôle découle de sa position et n’est jamais sélectionné séparément.
_Avoid_: Maître du jeu

**Équipe photographiée**:
L’équipe du responsable du score, dont les plis remportés sont photographiés après chaque manche.
_Avoid_: Notre équipe, équipe du téléphone, équipe Nous

**Score cible**:
Le seuil choisi au début de la partie qui en déclenche la fin; 2 000 points par défaut, avec 1 500 et 3 000 comme alternatives.

**Partie active**:
L’unique partie commencée mais non terminée, reprise automatiquement à sa dernière étape connue.
_Avoid_: Partie en cours

**Manche en préparation**:
La manche courante, sauvegardée mais encore modifiable, dont le résultat n’affecte pas encore les scores cumulés.
_Avoid_: Brouillon de manche

**Résultat provisoire**:
Le calcul vérifiable d’une manche en préparation, sans effet sur les scores cumulés tant que la manche n’est pas validée.
_Avoid_: Score temporaire

**Manche validée**:
Une manche confirmée dont le résultat contribue aux scores cumulés. Elle peut être dévalidée jusqu’au début de la manche suivante, puis devient immuable.
_Avoid_: Manche enregistrée

**Manche de départage**:
Une manche supplémentaire jouée lorsque les deux équipes atteignent ou dépassent le score cible avec un total identique. Les manches de départage continuent jusqu’à ce que les scores diffèrent.

**Annonce**:
Une suite ou un carré déclaré pendant la manche et susceptible d’accorder un bonus.
_Avoid_: Enchère

**Annonces gagnantes**:
L’ensemble des annonces conservées par l’unique équipe ayant remporté leur comparaison pendant une manche. L’autre équipe ne conserve aucune annonce.
_Avoid_: Primes, annonces des deux équipes

**Belote**:
Le bonus de 20 points associé au Roi et à la Dame d’une même couleur d’atout. Il n’existe pas en Sans Atout et peut être multiple en Tout Atout, une fois par couleur.
_Avoid_: Annonce

**Capot**:
Le résultat d’une manche dans laquelle une équipe remporte les huit plis, que ce résultat ait été annoncé ou non.

**Contrat Capot**:
Un contrat spécial qui impose à l’équipe preneuse de réaliser un Capot.
_Avoid_: Capot annoncé

**Générale**:
Un contrat spécial qui impose au joueur ayant enchéri de remporter personnellement les huit plis.

**Coinche**:
La contestation d’un adversaire qui double la valeur du contrat en cours.
_Avoid_: Contre

**Surcoinche**:
La réponse de l’équipe preneuse à une coinche, qui multiplie la valeur du contrat par quatre.
_Avoid_: Surcontre
