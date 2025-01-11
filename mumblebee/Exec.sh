#!/bin/bash

# Le script doit être exécuté avec les arguments suivants:
# $ ./do.sh <nx> <ny> [iter] [shared]
# avec:
# - nx: nombre de colonnes
# - ny: nombre de lignes
# - iter: nombre d'itérations (optionnel, par défaut 3000)
# - shared: 1 pour activer le partage de mémoire, 0 sinon (optionnel, par défaut 0)

make clean

shared=''
iter=3000  # Valeur par défaut pour le nombre d'itérations

# Vérifier si les arguments sont corrects
if [ "$#" -lt 2 ]; then
    echo "Erreur : Le script doit être exécuté avec les arguments suivants :"
    echo "  $ ./do.sh <nx> <ny> [iter] [shared]"
    echo "avec :"
    echo "  - nx : nombre de colonnes"
    echo "  - ny : nombre de lignes"
    echo "  - iter : nombre d'itérations (optionnel, par défaut 3000)"
    echo "  - shared : s pour activer le partage de mémoire"
    exit 1
else
    nx=$1
    ny=$2

    # Vérifier l'argument shared
    if [ "$#" -ge 3 ] && [ "$3" == "s" ]; then
        shared="--s"
        if [ "$#" -eq 4 ]; then
            iter="$4"
        fi
    elif [ "$#" -ge 3 ]; then
        iter="$3"
        if [ "$#" -eq 4 ] && [ "$4" == "s" ]; then
            shared="--s"
        fi
    fi
fi

# On make le programme
make

# Vérifier si le programme a bien été compilé
if [ ! -f "main" ]; then
    echo "Erreur : Le programme 'main' n'a pas été compilé."
    exit 1
fi

# Si les fichier 'out.txt', 'plot.gp' et 'out.png' existent, les supprimer
rm -f out.txt plot.gp out.png
touch out.txt plot.gp out.png

chmod 777 out.txt plot.gp out.png

# Afficher la commande à exécuter
echo "Commande : ./main --w=$nx --h=$ny --i=$iter $shared > out.txt"

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

# Créer le fichier de script Gnuplot dynamique
cat <<EOL > plot.gp
set terminal pngcairo size 800,600
set output 'out.png'
set pm3d map
set palette model RGB defined (0 "black", 1 "blue", 2 "green", 3 "yellow", 4 "red")
unset colorbox

# Ajuster la taille de l'affichage sans marges supplémentaires
set size ratio -1  # Force un ratio d'aspect exact (sans distorsion)

# Définir dynamiquement les limites en fonction des arguments
set xrange [0:$nx]   # Largeur de la matrice
set yrange [0:$ny]   # Hauteur de la matrice

# Tracer les données avec l'option 'matrix' pour ajuster les pixels précisément
splot 'out.txt' matrix with image
EOL

# Lancer le script Gnuplot
gnuplot plot.gp

# Vérifier si l'image a bien été générée
if [ ! -f "out.png" ]; then
    echo "Erreur : L'image 'out.png' n'a pas été générée."
    exit 1
fi

# Afficher l'image générée
xdg-open out.png

# Nettoyer les fichiers temporaires
rm -f out.txt plot.gp

# Fin du script
echo "Le script a terminé avec succès."
