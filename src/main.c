/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

/****************************************************/
#ifdef __APPLE__
	//nothing to define, do not have argp.h
#else
	#define HAVE_ARGP
#endif

/********************  HEADERS  *********************/
#include <math.h>
#include <mpi.h>
//if available use argp, if not fallback to getopt
#ifdef HAVE_ARGP
	#include <argp.h>
#else
	#include <getopt.h>
#endif
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <unistd.h>
#include <time.h>
#include "lbm_config.h"
#include "lbm_struct.h"
#include "lbm_phys.h"
#include "lbm_init.h"
#include "lbm_comm.h"
#include "lbm_save.h"
#include "exercises.h"

/****************************************************/
const char *argp_program_version = "lbm 1.0";
const char *argp_program_bug_address = "<sebastien.valat@univ-grenoble-alpes.fr>";
static char doc[] = "A simple simulation applying the Lattice Boltzmann Method (LBM)";
#ifdef HAVE_ARGP
	static char args_doc[] = "";
	static struct argp_option options[] = {
		{"config",   'c', "FILE",  0, "Input config file to use." },
		{"exercise", 'e', "EXID",  0, "ID of the exercice to execute." },
		{"no-out",   'n', 0,       0, "Skip output for benchmarking only compute and communications."},
		{"scaling",  's', "FACTOR",0, "Apply weak scaling factor to increase the mesh size."},
		{ 0 }
	};
#else
	static struct option long_options[] = {
			{ "config",     required_argument,      NULL,           'c' },
			{ "exercise",   required_argument,      NULL,           'e' },
			{ "scaling",    required_argument,      NULL,           's' },
			{ "no-out",     no_argument,            NULL,           'n' },
			{ "help",       no_argument,            NULL,           '?' },
			{ NULL,         0,                      NULL,           0 }
	};
	static const char * short_options = "[-c CONFIG] [-e EXID] [-s SCALE] [-n]";
	static const char * help_message = 
		"-c/--config   {FILE}    Input config file to use.\n"
		"-e/--exercise {EXID}    ID of the exercise to execute\n"
		"-n/--no-out             Skip output for benchmarking only compute and communications.\n"
		"-s/--scaling  {FACTOR}  Apply weak scaling factor to increase the mesh size.\n";
#endif

/****************************************************/
/* Used by main to communicate with parse_opt. */
struct arguments
{
	bool do_output;
	int exercice;
	char * config_file;
	int scaling;
};

/****************************************************/
/* Parse a single option. */
#ifdef HAVE_ARGP
static error_t parse_opt (int key, char *arg, struct argp_state *state)
{
	//get args
	struct arguments *arguments = state->input;

	switch (key)
	{
		case 'c':
			arguments->config_file = arg;
			break;
		case 'e':
			arguments->exercice = atoi(arg);
			break;
		case 's':
			arguments->scaling = atoi(arg);
			break;
		case 'n':
			arguments->do_output = false;
			break;
		case ARGP_KEY_ARG:
			argp_usage (state);
			break;
		case ARGP_KEY_END:
			break;
		default:
			return ARGP_ERR_UNKNOWN;
	}
	return 0;
}
#endif //HAVE_ARGP

/****************************************************/
/* Our argp parser. */
#ifdef HAVE_ARGP
	static struct argp argp = { options, parse_opt, args_doc, doc };
#endif //HAVE_ARGP

/****************************************************/
static inline double timespec_diff(struct timespec *a, struct timespec *b) {
        struct timespec result;
        result.tv_sec  = a->tv_sec  - b->tv_sec;
        result.tv_nsec = a->tv_nsec - b->tv_nsec;
        if (result.tv_nsec < 0) {
                --result.tv_sec;
                result.tv_nsec += 1000000000L;
        }
        return (double)result.tv_sec + (double)result.tv_nsec / (double)1e9;
}

/****************************************************/
#ifdef HAVE_ARGP
void parse_prgm_arguments(struct arguments * arguments, int argc, char ** argv)
{
	argp_parse (&argp, argc, argv, 0, 0, arguments);
}
#else //HAVE_ARGP
void print_help_message(char ** argv)
{
	printf("%s\n", argp_program_version);
	printf("%s\n", argp_program_bug_address);
	printf("%s\n", doc);
	printf("\n");
	printf("%s %s\n", argv[0], short_options);
	printf("\nOptions:\n");
	printf("%s", help_message);
}

void parse_prgm_arguments(struct arguments * arguments, int argc, char ** argv)
{
	int c;
	while ( (c = getopt_long(argc, argv, "c:e:s:nh", long_options, NULL)) != -1) {
		switch(c) {
			case 'c':
				arguments->config_file = strdup(optarg);
				break;
			case 'e':
				arguments->exercice = atoi(optarg);
				break;
			case 's':
				arguments->scaling = atoi(optarg);
				break;
			case 'n':
				arguments->do_output = false;
				break;
			case 'h':
			case '?':
				print_help_message(argv);
				exit(0);
				break;
			default:
				fatal("Invalid option !");
				break;
		}
	}

	//error
	if (optind != argc) {
		int index;
		for (index = optind; index < argc; index++)
			fprintf(stderr, "Invalid option : %s\n", argv[index]);
		printf("\n");
		exit(1);
	}
}
#endif //HAVE_ARGP

/****************************************************/
int main(int argc, char * argv[])
{
	//vars
	lbm_mesh_t mesh;
	lbm_mesh_t temp;
	lbm_mesh_type_t mesh_type;
	lbm_comm_t comm;
	lbm_file_mesh_t save_mesh;
	int i, rank, comm_size;
	const char * config_filename = NULL;

	//init MPI and get current rank and commuincator size.
	MPI_Init( &argc, &argv );
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

	//parse args
	struct arguments arguments = {
		.do_output = true,
		.exercice = 0,
		.config_file = "config.txt",
		.scaling = 1,
	};
	parse_prgm_arguments(&arguments, argc, argv);

	//dispatch
	config_filename = arguments.config_file;

	//load config file and display it on master node
	lbm_config_init(config_filename);

	//apply option
	if (arguments.do_output == false)
		RESULT_FILENAME = NULL;

	//apply scaling
	if (arguments.scaling > 1) {
		//apply
		double factor = sqrt((double)arguments.scaling);
		MESH_WIDTH = (double)MESH_WIDTH * factor;
		MESH_HEIGHT = (double)MESH_HEIGHT * factor;

		//make multiple of
		if (MESH_WIDTH % comm_size != 0)
			MESH_WIDTH += comm_size - MESH_WIDTH % comm_size;
		if (MESH_HEIGHT % comm_size != 0)
			MESH_HEIGHT -= MESH_HEIGHT % comm_size;

		//recompute obstable
		lbm_gbl_config.obstacle_r = (lbm_gbl_config.height / 10.0 + 1.0);
		lbm_gbl_config.obstacle_y = (lbm_gbl_config.height / 2.0 + 3.0);
	}

	//print config
	if (rank == RANK_MASTER)
		lbm_config_print();

	//dispatch
	lbm_ex_select(arguments.exercice);

	//init structures, allocate memory...
	lbm_comm_init_ex_select( &comm, MESH_WIDTH, MESH_HEIGHT);
	lbm_mesh_init( &mesh, lbm_comm_width( &comm ), lbm_comm_height( &comm ) );
	lbm_mesh_init( &temp, lbm_comm_width( &comm ), lbm_comm_height( &comm ) );
	lbm_mesh_type_t_init( &mesh_type, lbm_comm_width( &comm ), lbm_comm_height( &comm ));
	lbm_save_mesh_init(&save_mesh, &comm);

	//truncate file
	if (RESULT_FILENAME != NULL) {
		if (rank == RANK_MASTER)
			unlink(RESULT_FILENAME);
		usleep(1000);
		MPI_Barrier(MPI_COMM_WORLD);
	}

	//master open the output file
	//if( rank == RANK_MASTER )
	lbm_open_output_file(&comm);
	MPI_Barrier(MPI_COMM_WORLD);

	//setup initial conditions on mesh
	lbm_init_mesh_state( &mesh, &mesh_type, &comm);
	lbm_init_mesh_state( &temp, &mesh_type, &comm);

	//write initial condition in output file
	if (lbm_gbl_config.output_filename != NULL)
		lbm_save_ex_select(&save_mesh, &comm, &mesh, &mesh_type, 0 / WRITE_STEP_INTERVAL);

	//start time
	struct timespec start;
	clock_gettime(CLOCK_MONOTONIC, &start);

	//measure full time
	struct timespec full_start;
	clock_gettime(CLOCK_MONOTONIC, &full_start);

	//time steps
	for ( i = 1 ; i < ITERATIONS ; i++ )
	{
		//compute
		lbm_do_step_ex_select(&comm, &mesh_type, &mesh, &temp );

		//save step
		if ( i % WRITE_STEP_INTERVAL == 0 && lbm_gbl_config.output_filename != NULL )
			lbm_save_ex_select(&save_mesh, &comm, &mesh, &mesh_type, i / WRITE_STEP_INTERVAL);
		
		//print progress
		if( rank == RANK_MASTER && i % WRITE_STEP_INTERVAL == 0 ) {
			//compute delta
			struct timespec stop;
			clock_gettime(CLOCK_MONOTONIC, &stop);
			double result = timespec_diff(&stop, &start);

			//printf
			printf("Progress [%5d / %5d] (%g s)\n",i,ITERATIONS, result);

			//copy back
			start = stop;
		}
	}

	//compute delta
	struct timespec full_stop;
	clock_gettime(CLOCK_MONOTONIC, &full_stop);
	double full_time = timespec_diff(&full_stop, &full_start);
	if (rank == 0)
		printf("Total time: %g seconds\n", full_time);

	//close file
	if (RESULT_FILENAME != NULL)
		MPI_File_close(&comm.file_handler);

	//free memory
	lbm_comm_release_ex_select( &comm );
	lbm_mesh_release( &mesh );
	lbm_mesh_release( &temp );
	lbm_mesh_type_t_release( &mesh_type );
	lbm_save_mesh_release(&save_mesh);

	//close MPI
	MPI_Finalize();

	return EXIT_SUCCESS;
}
