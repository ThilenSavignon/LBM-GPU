#!/bin/bash

# Warning message the script is long



make clean

shared=''
iter=1  # Initial number of iterations
num_iterations=25  # Number of iterations

if ! command -v gnuplot &> /dev/null; then
    echo "Error: Gnuplot is not installed. Please install it to run this script."
    exit 1
fi

# Vérifier si les arguments sont corrects
if [ "$#" -lt 2 ]; then
    echo "Error: The script must be executed with the following arguments:"
    echo "  $ ./do.sh <nx> <ny> [shared]"
    echo "with:"
    echo "  - nx : number of columns"
    echo "  - ny : number of rows"
    echo '  - shared : "s" to enable shared memory (optional)'
    exit 1
else
    nx=$1
    ny=$2

    # Vérifier l'argument 'shared'
    if [ "$#" -ge 3 ] && [ "$3" == "s" ]; then
        shared="--s"
    fi
fi

# We clean the project
make clean

# We compile the project
make

# Verify if the program was compiled correctly
if [ ! -f "main" ]; then
    echo "Error: The 'main' program was not compiled."
    exit 1
fi

# if the 'gif' folder exists, delete it and recreate it
if [ -d "gif" ]; then
    rm -rf gif
fi
mkdir gif

# Remove the temporary files
rm -f out.txt plot.gp out.png

# Run the program for each iteration
for i in $(seq 1 $num_iterations); do
    # iter become 100 * iter²
    iter=$((100 * (i) * (i)))

    # Display the command executed
    echo "Execution $((i)) : Command : ./main --w=$nx --h=$ny --i=$iter $shared > out.txt"

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

    # Create the Gnuplot script
    cat <<EOL > plot.gp
set terminal pngcairo size 800,600
set output "gif/out_$((i+1)).png"
set pm3d map
set palette model RGB defined (0 "black", 1 "blue", 2 "green", 3 "yellow", 4 "red")
unset colorbox

# Adjust the aspect ratio
set size ratio -1

# Define the range of the plot
set xrange [0:$nx]   # Largeur de la matrice
set yrange [0:$ny]   # Hauteur de la matrice

# Set the title of the plot
set label "out$((i)).png" at graph 0.5, 1.1 center offset 0, 0.05 font ",14"

# Plot the data from the 'out.txt' file (excluding the last line)
splot '< tail -n +1 out.txt | head -n -1' matrix with image
EOL

    # Run the Gnuplot script
    gnuplot plot.gp

    # Verify if the 'out.png' image was generated
    if [ ! -f "gif/out_$((i+1)).png" ]; then
        echo "Error: The image 'gif/out_$((i+1)).png' was not generated."
        exit 1
    fi
done

# Create the GIF from the images
magick -delay 20 -loop 0 gif/out_*.png gif/output.gif

# Verify if the GIF was generated
if [ ! -f "gif/output.gif" ]; then
    echo "Error: The GIF 'gif/output.gif' was not generated."
    exit 1
fi

# Display the GIF
xdg-open gif/output.gif

# Remove the temporary files
rm -f out.txt plot.gp

# Display the success message
echo "Script ended successfully."
