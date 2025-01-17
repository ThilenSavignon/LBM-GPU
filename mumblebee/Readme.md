Running the model
=================

```sh
srun 
```

Objectif restant
================

- Simulation sous les différents contextes.
- Créer des courbes de performances: nb de blocks, tuiles, taille matrice
- Comparatif cpu, thread/non-thread et gpu
- Créer une version shared memory
- Sécurisé l'itération en cours pour lancer la suivante sur les thread en attente.
- passer l'initialisation dans le cuda
  - Créer/allocer des variables de réception
  - (f, feq, rho, ux , uy, usqr, mesh)
  - (DR, WALL, FL)