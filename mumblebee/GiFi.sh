#!/bin/bash

# Le script doit être exécuté avec les arguments suivants:
# $ ./do.sh <nx> <ny> [shared]
# avec:
# - nx: nombre de colonnes
# - ny: nombre de lignes
# - shared : 1 pour activer le partage de mémoire, 0 sinon (optionnel)

make clean

shared=''
iter=1  # Nombre d'itérations de départ
num_iterations=25  # Nombre d'exécutions

# Vérifier si les arguments sont corrects
if [ "$#" -lt 2 ]; then
    echo "Erreur : Le script doit être exécuté avec les arguments suivants :"
    echo "  $ ./do.sh <nx> <ny> [shared]"
    echo "avec :"
    echo "  - nx : nombre de colonnes"
    echo "  - ny : nombre de lignes"
    echo "  - shared : s pour activer le partage de mémoire"
    exit 1
else
    nx=$1
    ny=$2

    # Vérifier l'argument shared
    if [ "$#" -ge 3 ] && [ "$3" == "s" ]; then
        shared="--s"
    fi
fi

# On fait un make clean pour nettoyer les anciennes compilations
make clean

# On make le programme
make

# Vérifier si le programme a bien été compilé
if [ ! -f "main" ]; then
    echo "Erreur : Le programme 'main' n'a pas été compilé."
    exit 1
fi

# Si le dossier 'gif' existe, le supprimer et le recréer
if [ -d "gif" ]; then
    rm -rf gif
fi
mkdir gif

# Si les fichiers 'out.txt', 'plot.gp' et 'out.png' existent, les supprimer
rm -f out.txt plot.gp out.png

# Boucle pour exécuter le programme 16 fois avec des itérations croissantes
for i in $(seq 1 $num_iterations); do
    # iter devient 100 * iter²
    iter=$((100 * (i) * (i)))

    # Afficher la commande à exécuter
    echo "Exécution $((i+1)) : Commande : ./main --w=$nx --h=$ny --i=$iter $shared > out.txt"

    # Lancer le programme principal
    if ! ./main --w=$nx --h=$ny --i=$iter $shared > out.txt; then
        echo "Erreur : Le programme 'main' n'a pas été exécuté correctement."
        exit 1
    fi

    # Vérifier si le programme a bien généré le fichier 'out.txt'
    if [ ! -f "out.txt" ]; then
        echo "Erreur : Le fichier 'out.txt' n'a pas été généré."
        exit 1
    fi

    # Créer le fichier de script Gnuplot dynamique pour chaque exécution
    cat <<EOL > plot.gp
set terminal pngcairo size 800,600
set output "gif/out_$((i+1)).png"
set pm3d map
set palette model RGB defined (0 "black", 1 "blue", 2 "green", 3 "yellow", 4 "red")
unset colorbox

# Ajuster la taille de l'affichage sans marges supplémentaires
set size ratio -1  # Force un ratio d'aspect exact (sans distorsion)

# Définir dynamiquement les limites en fonction des arguments
set xrange [0:$nx]   # Largeur de la matrice
set yrange [0:$ny]   # Hauteur de la matrice

# Ajouter un label avec le numéro de l'itération
set label "Itération $((i))" at graph 0.5, 1.1 center offset 0, 0.05 font ",14"

# Tracer les données avec l'option 'matrix' pour ajuster les pixels précisément
splot 'out.txt' matrix with image
EOL

    # Lancer le script Gnuplot pour chaque exécution
    gnuplot plot.gp

    # Vérifier si l'image a bien été générée
    if [ ! -f "gif/out_$((i+1)).png" ]; then
        echo "Erreur : L'image 'gif/out_$((i+1)).png' n'a pas été générée."
        exit 1
    fi
done

# Créer le GIF avec toutes les images générées
magick -delay 20 -loop 0 gif/out_*.png gif/output.gif

# Vérifier si le GIF a bien été généré
if [ ! -f "gif/output.gif" ]; then
    echo "Erreur : Le GIF 'gif/output.gif' n'a pas été généré."
    exit 1
fi

# Afficher le GIF généré
xdg-open gif/output.gif

# Nettoyer les fichiers temporaires
rm -f out.txt plot.gp

# Fin du script
echo "Le script a terminé avec succès."
