#!/bin/bash

set -e

function bench()
{
	echo "#processes ex1 ex2 ex3 ex4 ex5 ex6" > scale.dat
	for scale in 1 2 4 8
	do
		echo > /tmp/output.dat
		for ex in 1 2 3 4 5 6
		do
			/usr/bin/time -f '%e' --append -o /tmp/output.dat mpirun --use-hwthread-cpus -np ${scale} ./lbm -n -e ${ex}
		done
		cat /tmp/output.dat | xargs echo ${scale} | tee -a scale.dat
	done
}

function plot()
{
ref=$(cat scale.dat | egrep '^1 ' | cut -f 2 -d ' ')
cat << EOF | gnuplot
	set term pdf
	set output "scale.pdf"
	set grid
	set key left
	set xlabel "MPI Tasks"
	set ylabel "Execution time (s)"
	plot "scale.dat" u 1:2 w lp title "linear split", "" u 1:3 w lp title "odd even", "" u 1:4 w lp title "async", "" u 1:5 w lp title "9", "" u 1:6 w lp title "9 mpi types", "" u 1:7 w lp title "overlap"

	plot "scale.dat" u 1:(\$2/$ref/\$1) w lp title "linear split", "" u 1:(\$3/$ref/\$1) w lp title "odd even", "" u 1:(\$4/$ref/\$1) w lp title "async", "" u 1:(\$5/$ref/\$1) w lp title "9", "" u 1:(\$6/$ref/\$1) w lp title "9 mpi types", "" u 1:(\$7/$ref/\$1) w lp title "overlap"
	
	set logscale y 2
	set key bottom
	plot "scale.dat" u 1:($ref/\$2) w lp title "linear split", "" u 1:($ref/\$3) w lp title "odd even", "" u 1:($ref/\$4) w lp title "async", "" u 1:($ref/\$5) w lp title "9", "" u 1:($ref/\$6) w lp title "9 mpi types", "" u 1:($ref/\$7) w lp title "overlap"
EOF
}

#bench
plot
rm -f /tmp/output.dat
