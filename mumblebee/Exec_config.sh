#!/bin/bash

make clean

shared=''
iter=3000  # Default number of iterations

# Verify if the arguments are correct
if [ "$#" -lt 2 ]; then
    echo "Error: The script must be executed with the following arguments:"
    echo "  $ ./do.sh <config_file> [iter] [shared]"
    echo "with:"
    echo "  - config_file : is the a file containing the number of columns and rows at the first line and the configuration of the matrix at the following lines"
    echo "  - iter : number of iterations (optional)"
    echo '  - shared : "s" to enable shared memory (optional)'
    exit 1
fi
config=$1

# Verify the 'iter' and 'shared' arguments
if [ "$#" -ge 2 ] && [ "$2" == "s" ]; then
    shared="--s"
    if [ "$#" -eq 3 ]; then
        iter="$3"
    fi
elif [ "$#" -ge 2 ]; then
    iter="$2"
    if [ "$#" -eq 3 ] && [ "$3" == "s" ]; then
        shared="--s"
    fi
fi

# Make the program
make

# Verify if the program was compiled correctly
if [ ! -f "main" ]; then
    echo "Error: The 'main' program was not compiled."
    exit 1
fi

# If the 'gif' folder exists, delete it and recreate it
rm -f out.txt plot.gp out.png
touch out.txt plot.gp out.png

chmod 777 out.txt plot.gp out.png

# Print the command to be executed
echo "Command : ./main --f=$config --i=$iter $shared > out.txt"

# Run the 'main' program
if ! ./main --f=$config --i=$iter $shared > out.txt; then
    echo "Error: The 'main' program failed to run."
    exit 1
fi

# Verify if the 'out.txt' file was generated
if [ ! -f "out.txt" ]; then
    echo "Error: The 'out.txt' file was not generated."
    exit 1
fi

read nx ny < $config
echo "$nx $ny"

# Create a Gnuplot script to plot the data
cat <<EOL > plot.gp
set terminal pngcairo size 800,600
set output 'out.png'
set pm3d map
set palette model RGB defined (0 "black", 1 "blue", 2 "green", 3 "yellow", 4 "red")
unset colorbox

# Fix the aspect ratio of the plot
set size ratio -1 

# Define the range of the plot
set xrange [0:$nx]   # Largeur de la matrice
set yrange [0:$ny]   # Hauteur de la matrice

# Title of the plot
splot 'out.txt' matrix with image
EOL

# Run Gnuplot to generate the image
gnuplot plot.gp

# Verify if the 'out.png' file was generated
if [ ! -f "out.png" ]; then
    echo "Error: The 'out.png' file was not generated."
    exit 1
fi

# Open the image with the default image viewer
xdg-open out.png

# Clean up the files
rm -f out.txt plot.gp

# Print a success message
echo "Success: The program was executed successfully."
