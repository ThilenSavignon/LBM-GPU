#!/bin/bash
#######################################################
#    AUTHOR  : Sébastien Valat                        #
#    MAIL    : sebastien.valat@univ-grenoble-alpes.fr #
#    LICENSE : BSD                                    #
#    YEAR    : 2021                                   #
#    COURSE  : Parallel Algorithms and Programming    #
#######################################################

if [ -z "$1" ]; then
	INPUT_FILE="output.raw"
else
	INPUT_FILE="$1"
fi
echo "INPUT_FILE: $INPUT_FILE"

if [ -z "$2" ]; then
	OUTPUT_FILE="${INPUT_FILE%.raw}.gif"
else
	OUTPUT_FILE="$2"
fi
echo "OUTPUT_FILE: $OUTPUT_FILE"

if [ "$INPUT_FILE" == "$OUTPUT_FILE" ]; then
	echo "Input and Output file are the same"
	echo "Usage : $0 [{input.raw}] [{output.gif}]" 1>&2
	exit 1
fi


#gen gnuplot command
gen_gnuplot_command_seq()
{
	# configure
	echo "set pm3d map"
	echo "set palette defined ( 0 '#000090', 1 '#000fff',2 '#0090ff',3 '#0fffee', 4 '#90ff70', 5 '#ffee00', 6 '#ff7000', 7 '#ee0000', 8 '#7f0000')"
	echo "set cbr [0:0.14]"
	echo "set term gif animate delay 5 size ${FINAL_WIDTH},${FINAL_HEIGHT}"
	echo "set output '${OUTPUT_FILE}'"

	# loop on frames
	local last_frame=$((${FRAMES} - 1))
	for tmp in $(seq 0 1 ${last_frame})
	do
		echo "print '$tmp / $((${FRAMES} - 1))'"
		echo "splot \"< ./display --gnuplot ${INPUT_FILE} $tmp\" u 1:2:4"
	done
}

#call if
run_seq()
{
	gen_gnuplot_command_seq | gnuplot
}

# fail
fail()
{
	echo "$@"
	exit 1
}


# compute draw size
calc_draw_size()
{
	# get local values
	local width=${WIDTH}
	local height=${HEIGHT}

	# calc image size
	FINAL_WIDTH=$((${width}+177+176))
	FINAL_HEIGHT=$((${height}+51+70))

	# reduce if too large
	while [[ ${FINAL_WIDTH} -gt 1920 || ${FINAL_HEIGHT} -gt 1080 ]]; do
		width=$((${width}/2))
		height=$((${height}/2))
		FINAL_WIDTH=$((${width}+177+176))
		FINAL_HEIGHT=$((${height}+51+70))
	done

	# avoid too small
	if [[ ${FINAL_WIDTH} < 600 || ${FINAL_HEIGHT} < 420 ]]; then
		width=$((${width}*2))
		height=$((${height}*2))
		FINAL_WIDTH=$((${width}+177+176))
		FINAL_HEIGHT=$((${height}+51+70))
	fi

	#echo ${FINAL_WIDTH}x${FINAL_HEIGHT} 1>&2
}

#gen gnuplot command
gen_gnuplot_command_parallel()
{
	# plot frame
	ID=$1
	echo "set pm3d map"
	echo "set palette defined ( 0 '#000090', 1 '#000fff',2 '#0090ff',3 '#0fffee', 4 '#90ff70', 5 '#ffee00', 6 '#ff7000', 7 '#ee0000', 8 '#7f0000')"
	echo "set cbr [0:0.14]"
	echo "set terminal png size ${FINAL_WIDTH},${FINAL_HEIGHT}"
	printf "set output '${TMPDIR}/%05d.png'\n" ${ID}
	echo "splot \"< ./display --gnuplot ${INPUT_FILE} ${ID} \" u 1:2:4"
}

run_parallel()
{
	#create tempdir
	TMPDIR='/tmp/simu-simple-lbm-gif-gen'
	rm -rf ${TMPDIR}
	mkdir ${TMPDIR}

	#call if
	last_frame=$((${FRAMES} - 1))
	for tmp in $(seq 0 1 ${last_frame})
	do
		echo "Image $tmp"
		gen_gnuplot_command_parallel $tmp > ${TMPDIR}/${tmp}.gnuplot
	done

	# build all in parallel
	parallel --bar gnuplot {} ::: ${TMPDIR}/*.gnuplot

	# merge all
	echo "Merger images..."
	convert -monitor ${TMPDIR}/*.png ${OUTPUT_FILE}

	# clean
	rm -rf ${TMPDIR}
}

# check
which gnuplot > /dev/null || fail "ERROR: You need GNUPLOT to make rendering !"
which parallel > /dev/null && HAVE_PARALLEL='yes' || HAVE_PARALLEL='no'
which convert > /dev/null && HAVE_CONVERT='yes' || HAVE_CONVERT='no'

#get info
./display --info ${INPUT_FILE} 0 > /dev/null || fail "Fail to read data file ${INPUT_FILE} !"
FRAMES=$(./display --info ${INPUT_FILE} 0 | grep frames | sed -e 's/frames=//g')
WIDTH=$(./display --info ${INPUT_FILE} 0 | grep width | sed -e 's/width=//g')
HEIGHT=$(./display --info ${INPUT_FILE} 0 | grep height | sed -e 's/height=//g')

#calc size
calc_draw_size

# select
if [[ ${HAVE_PARALLEL} == 'yes' && ${HAVE_CONVERT} == 'yes' ]]; then
	run_parallel
else
	echo "Warning: Not have GNU PARALLEL insalled, fallback to sequential version !"
	run_seq
	echo "Warning: Not have GNU PARALLEL insalled, fallback to sequential version !"
fi

# ok
echo "Done"
