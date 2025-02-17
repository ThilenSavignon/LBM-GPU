#!/bin/bash

make clean

shared=''
iter=3000  # Default number of iterations

if ! command -v gnuplot &> /dev/null; then
    echo "Error: Gnuplot is not installed. Please install it to run this script."
    exit 1
fi

# Verify if the arguments are correct
if [ "$#" -lt 2 ]; then
    echo "Error: The script must be executed with the following arguments:"
    echo "  $ ./do.sh <nx> <ny> [iter] [shared]"
    echo "with:"
    echo "  - nx : number of columns"
    echo "  - ny : number of rows"
    echo "  - iter : number of iterations (optional)"
    echo '  - shared : "s" to enable shared memory (optional)'
    exit 1
else
    nx=$1
    ny=$2

    # Verify the 'iter' and 'shared' arguments
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

# Plot the data from the 'out.txt' file (excluding the last line)
echo "Command : ./main --w=$nx --h=$ny --i=$iter $shared > out.txt"

# Run the 'main' program
if ! ./main --w=$nx --h=$ny --i=$iter $shared > out.txt; then
    echo "Error: The 'main' program was not executed correctly."
    exit 1
fi

# Verify if the 'out.txt' file was generated
if [ ! -f "out.txt" ]; then
    echo "Error: The 'out.txt' file was not generated."
    exit 1
fi

# Get the last line of the output
last_line=$(tail -n 2 out.txt)


# Create a temporary file with all but the last line of out.txt
head -n -1 out.txt > temp_out.txt

# Create a Gnuplot script to plot the data from the temporary file
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

# Plot the data from the temporary file
splot 'temp_out.txt' matrix with image
EOL

# Run the Gnuplot script
gnuplot plot.gp

# Verify if the 'out.png' image was generated
if [ ! -f "out.png" ]; then
    echo "Error: The image 'out.png' was not generated."
    exit 1
fi

# Display the generated image
xdg-open out.png

# Clean up the files
rm -f out.txt plot.gp temp_out.txt

# Display the last line of the output
echo "Script ended successfully."
echo $last_line
