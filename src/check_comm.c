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
#ifdef HAVE_ARGP
	#include <argp.h>
#else
	#include <unistd.h>
	#include <getopt.h>
#endif
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdbool.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "lbm_config.h"
#include "lbm_struct.h"
#include "lbm_phys.h"
#include "lbm_init.h"
#include "lbm_comm.h"
#include "exercises.h"

/****************************************************/
#ifdef DISABLE_COLORS
	#define RESET   ""
	#define RED     ""
	#define GREEN  ""
	#define YELLOW  ""
	#define GREY  ""
#else
	/** Reset colors in terminal. **/
	#define RESET   "\033[0m"
	/** Make colors in red **/
	#define RED     "\033[31m"
	/** Make colors in red **/
	#define GREEN  "\033[32m"
	/** Make colors in red **/
	#define YELLOW  "\033[33m"
	/** Make colors in red **/
	#define GREY  "\033[90m"
#endif

/****************************************************/
/** Define coored. **/
typedef int lbm_coords_t[2];

/****************************************************/
typedef enum lbm_show_mode_e {
	LBM_SHOW_CURRENT,
	LBM_SHOW_EXPECTED,
	LBM_SHOW_BOTH,
} lbm_show_mode_t;

/****************************************************/
typedef enum lbm_fill_mode_s {
	LBM_FILL_RANK,
	LBM_FILL_MODULO_9,
	LBM_FILL_MODULO_10,
	LBM_FILL_POSITION,
} lbm_fill_mode_t;

/****************************************************/
// To parse arguments.
const char *argp_program_version = "check_comm 1.0";
const char *argp_program_bug_address = "<sebastien.valat@univ-grenoble-alpes.fr>";
static char doc[] = "A simple checker for communication developpement for LBM.";
#ifdef HAVE_ARGP
	static char args_doc[] = "";
	static struct argp_option options[] = {
		{"width",         'w', "WIDTH",  0, "Total width of the mesh to compute and print." },
		{"height",        'h', "HEIGHT", 0, "Total height of the mesh to compute and print." },
		{"exercise",      'e', "EXID",   0, "ID of the exercice to execute." },
		{"show",          's', "MODE",   0, "Show expected value on error: 'current', 'expected', or 'both'"},
		{"pattern",       'p', "PATTERN",0, "Define how to fill the mesh: 'rank', 'modulo9', 'modulo10' or 'position'."},
		{ 0 }
	};
#else
	static struct option long_options[] = {
			{ "width",      required_argument,      NULL,           'w' },
			{ "height",     required_argument,      NULL,           'h' },
			{ "exercise",   required_argument,      NULL,           'e' },
			{ "show",       required_argument,      NULL,           's' },
			{ "pattern",    required_argument,      NULL,           'p' },
			{ "help",       no_argument,            NULL,           '?' },
			{ NULL,         0,                      NULL,           0 }
	};
	static const char * short_options = "[-w WIDTH] [-h HEIGHT] [-e EXID] [-s MODE] [-p PATTERN]";
	static const char * help_message = 
		"-w/--with     {WIDTH}   Total width of the mesh to compute and print.\n"
		"-h/--height   {HEIGHT}  Total height of the mesh to compute and print.\n"
		"-e/--exercise {EXID}    ID of the exercise to execute\n"
		"-s/--show     {MODE}    Show expected value on error: 'current', 'expected', or 'both'.\n"
		"-p/--pattern  {PATTERN} Define how to fill the mesh: 'rank', 'modulo9', 'modulo10' or 'position'.\n";
#endif

/****************************************************/
/* Used by main to communicate with parse_opt. */
struct arguments
{
	int exercice;
	int width;
	int height;
	lbm_show_mode_t show;
	lbm_fill_mode_t fill;
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
		case 'w':
			arguments->width = atoi(arg);
			break;
		case 'h':
			arguments->height = atoi(arg);
			break;
		case 'e':
			arguments->exercice = atoi(arg);
			break;
		case 'p':
			if (strcmp(arg, "rank") == 0)
				arguments->fill = LBM_FILL_RANK;
			else if (strcmp(arg, "modulo9") == 0)
				arguments->fill = LBM_FILL_MODULO_9;
			else if (strcmp(arg, "modulo10") == 0)
				arguments->fill = LBM_FILL_MODULO_10;
			else if (strcmp(arg, "position") == 0)
				arguments->fill = LBM_FILL_POSITION;
			else
				fatal("Invalid value for -m/--mode option !");
			break;
		case 's':
			if (strcmp(arg, "current") == 0)
				arguments->show = LBM_SHOW_CURRENT;
			else if (strcmp(arg, "expected") == 0)
				arguments->show = LBM_SHOW_EXPECTED;
			else if (strcmp(arg, "both") == 0)
				arguments->show = LBM_SHOW_BOTH;
			else
				fatal("Invalid value for -s/--show option !");
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
#endif

/****************************************************/
/* Our argp parser. */
#ifdef HAVE_ARGP
	static struct argp argp = { options, parse_opt, args_doc, doc };
#endif

/****************************************************/
int get_rank_at(lbm_comm_t * comm, lbm_coords_t * coords, int world_size, int x, int y, int outvalue)
{
	//outside
	if (x < 0 || y < 0)
		return outvalue;
	if (x >= comm->nb_x || y >= comm->nb_y)
		return outvalue;
	
	//search
	int rank;
	for (rank = 0 ; rank < world_size ; rank++)
		if (coords[rank][0] == x && coords[rank][1] == y)
			break;

	//return
	return rank;
}

/****************************************************/
bool is_on_border(lbm_comm_t * comm, int col, int line)
{
	return (col == 0 || col == comm->width -1 || line == 0 || line == comm->height - 1);
}

/****************************************************/
int calc_expected_value(lbm_comm_t * comm, lbm_coords_t * coords, int x, int y, int col, int line, int rank, int world_size, lbm_fill_mode_t fill)
{
	//easy case
	if (fill == LBM_FILL_POSITION)
		return ((x * (comm->width-2) + col) * (comm->height) + (y * (comm->height-2)) + line);
	else if (fill == LBM_FILL_MODULO_9)
		return ((x * (comm->width-2) + col) * (comm->height) + (y * (comm->height-2)) + line)%9;
	else if (fill == LBM_FILL_MODULO_10)
		return ((x * (comm->width-2) + col) * (comm->height) + (y * (comm->height-2)) + line)%10;

	//calc expect location
	int dx = 0;
	int dy = 0;

	//for borders
	if (col == 0) dx = -1;
	if (line == 0 && comm->nb_y > 1) dy = -1;
	if (col == comm->width-1) dx=+1;
	if (line == comm->height-1 && comm->nb_y > 1) dy=+1;

	//get expect
	int expect1 = get_rank_at(comm, coords, world_size, x+dx, y+dy, -1);
	int expect2 = get_rank_at(comm, coords, world_size, x+dx, y, -1);
	int expect3 = get_rank_at(comm, coords, world_size, x, y+dy, -1);

	//return the one not -1 or rank
	if (expect1 != -1) return expect1;
	if (expect2 != -1) return expect2;
	if (expect3 != -1) return expect3;
	return rank;
}

/****************************************************/
bool check_cell_values(char * our_error_msg, lbm_mesh_cell_t cell, int x, int y, int col, int line)
{
	//vars
	int k;
	int value = (int)cell[0];

	//check inner
	for (k = 0 ; k < DIRECTIONS ; k++) {
		if ((int)cell[k] != value) {
			//fill message
			sprintf(our_error_msg, YELLOW "Error inside cell, all 9 direction not valid (should be same): rank=(%d, %d) coord=(%d, %d)\ncell=[%f, %f, %f, %f, %f, %f, %f, %f, %f]" RESET "\n",
				x, y,
				col, line,
				cell[0], cell[1], cell[2],
				cell[3], cell[4], cell[5],
				cell[6], cell[7], cell[8]);
			
			//not ok
			return false;
		}
	}

	//ok
	return true;
}

void disaply_value(int value, int expected, bool is_border, bool err_cell, lbm_show_mode_t show, lbm_fill_mode_t fill)
{
	//display value
	int display = value;
	if (show== LBM_SHOW_EXPECTED)
		display = expected;

	if (fill == LBM_FILL_POSITION) {
		if (value != expected) {
			if (show == LBM_SHOW_BOTH)
				printf(RED "%03d" GREEN "%03d" RESET " ", display, expected);
			else
				printf(RED "%03d" RESET " ", display);
		} else if (err_cell) {
			if (show == LBM_SHOW_BOTH)
				printf(YELLOW "%03d" RESET "    ", display);
			else
				printf(YELLOW "%03d" RESET " ", display);
		} else if (is_border) {
			if (show == LBM_SHOW_BOTH)
				printf(GREY "%03d" RESET "    ", display);
			else
				printf(GREY "%03d" RESET " ", display);
		} else {
			if (show == LBM_SHOW_BOTH)
				printf("%03d    ", display);
			else
				printf("%03d ", display);
		}
	} else {
		if (value != expected) {
			if (show == LBM_SHOW_BOTH)
				printf(RED "%d" GREEN "%d" RESET " ", display, expected);
			else
				printf(RED "%d" RESET " ", display);
		} else if (err_cell) {
			if (show == LBM_SHOW_BOTH)
				printf(YELLOW "%d" RESET "  ", display);
			else
				printf(YELLOW "%d" RESET " ", display);
		} else if (is_border) {
			if (show == LBM_SHOW_BOTH)
				printf(GREY "%d" RESET "  ", display);
			else
				printf(GREY "%d" RESET " ", display);
		} else {
			if (show == LBM_SHOW_BOTH)
				printf("%d  ", display);
			else
				printf("%d ", display);
		}
	}
}

/****************************************************/
void display_meshes(lbm_comm_t * comm, lbm_mesh_t * mesh_rank, lbm_coords_t * coords, int world_size, lbm_show_mode_t show, lbm_fill_mode_t fill)
{
	bool ok = true;
	int x,y,col,line;
	char dim_error[1024] = "";
	//loop all meshes
	for (y = 0 ; y < comm->nb_y ; y++) {
		for (line = 0 ; line < comm->height ; line++) {
			for (x = 0 ; x < comm->nb_x ; x++) {
				//found corresponding rank
				int rank = get_rank_at(comm, coords, world_size, x, y, -1);
				assert(rank != -1);

				//display mesh
				for (col = 0 ; col < comm->width ; col++) {
					//extract cell
					lbm_mesh_cell_t cell = lbm_mesh_get_cell(&mesh_rank[rank], col, line);
					int value = (int)cell[0];

					//check
					bool is_border = is_on_border(comm, col, line);
					int expected = calc_expected_value(comm, coords, x, y, col, line, rank, world_size, fill);
					bool err_cell = !check_cell_values(dim_error, cell, x, y, col, line);

					//display
					disaply_value(value, expected, is_border, err_cell, show, fill);

					//final status
					if (value != expected || err_cell)
						ok = false;
				}
				printf("  ");
			}
			printf("\n");
		}
		printf("\n");
	}

	//print possible inside error
	fprintf(stderr, "%s", dim_error);

	if (ok)
		printf(GREEN ">>>  VALID  <<<" RESET "\n");
	else
		fprintf(stderr, RED ">>>  ERROR  <<<" RESET "\n");
}

/****************************************************/
void mesh_init_rank(lbm_mesh_t * mesh, int rank)
{
	//vars
	int i,j,k;

	//errors
	assert(mesh != NULL);

	//loop on all cells
	for ( i = 0 ; i <  mesh->width ; i++)
		for ( j = 0 ; j <  mesh->height ; j++)
			for ( k = 0 ; k < DIRECTIONS ; k++)
				lbm_mesh_get_cell(mesh, i, j)[k] = rank;
}

/****************************************************/
void mesh_init_zero_ghost(lbm_mesh_t * mesh, lbm_comm_t * comm)
{
	//vars
	int i,j,k;

	//border
	if (comm->x - 1 > 0)
		for ( j = 0 ; j <  mesh->height ; j++)
			for ( k = 0 ; k < DIRECTIONS ; k++)
				lbm_mesh_get_cell(mesh, 0, j)[k] = 0;
	if (comm->x + 1 < comm->nb_x)
		for ( j = 0 ; j <  mesh->height ; j++)
			for ( k = 0 ; k < DIRECTIONS ; k++)
				lbm_mesh_get_cell(mesh, mesh->width-1, j)[k] = 0;
	if (comm->y - 1 > 0)
		for ( i = 0 ; i <  mesh->width ; i++)
			for ( k = 0 ; k < DIRECTIONS ; k++)
				lbm_mesh_get_cell(mesh, i, 0)[k] = 0;
	if (comm->y + 1 < comm->nb_y)
		for ( i = 0 ; i <  mesh->width ; i++)
			for ( k = 0 ; k < DIRECTIONS ; k++)
				lbm_mesh_get_cell(mesh, i, mesh->height-1)[k] = 0;
}

/****************************************************/
void mesh_init_mod(lbm_mesh_t * mesh, lbm_comm_t * comm, int modulo)
{
	//vars
	int i,j,k;

	//errors
	assert(mesh != NULL);

	//loop on all cells
	for ( i = 0 ; i <  mesh->width ; i++)
		for ( j = 0 ; j <  mesh->height ; j++)
			for ( k = 0 ; k < DIRECTIONS ; k++)
				lbm_mesh_get_cell(mesh, i, j)[k] = ((comm->x + i) * mesh->height + comm->y + j)%modulo;

	//zero ghost
	mesh_init_zero_ghost(mesh, comm);
}

/****************************************************/
void mesh_init_pos(lbm_mesh_t * mesh, lbm_comm_t * comm)
{
	//vars
	int i,j,k;

	//errors
	assert(mesh != NULL);

	//loop on all cells
	for ( i = 0 ; i <  mesh->width ; i++)
		for ( j = 0 ; j <  mesh->height ; j++)
			for ( k = 0 ; k < DIRECTIONS ; k++)
				lbm_mesh_get_cell(mesh, i, j)[k] = ((comm->x + i) * mesh->height + comm->y + j);

	//zero ghost
	mesh_init_zero_ghost(mesh, comm);
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
	while ( (c = getopt_long(argc, argv, "w:h:e:s:p:", long_options, NULL)) != -1) {
		switch(c) {
			case 'w':
				arguments->width = atoi(optarg);
				break;
			case 'h':
				arguments->height = atoi(optarg);
				break;
			case 'e':
				arguments->exercice = atoi(optarg);
				break;
			case 'p':
				if (strcmp(optarg, "rank") == 0)
					arguments->fill = LBM_FILL_RANK;
				else if (strcmp(optarg, "modulo9") == 0)
					arguments->fill = LBM_FILL_MODULO_9;
				else if (strcmp(optarg, "modulo10") == 0)
					arguments->fill = LBM_FILL_MODULO_10;
				else if (strcmp(optarg, "position") == 0)
					arguments->fill = LBM_FILL_POSITION;
				else
					fatal("Invalid value for -m/--mode option !");
				break;
			case 's':
				if (strcmp(optarg, "current") == 0)
					arguments->show = LBM_SHOW_CURRENT;
				else if (strcmp(optarg, "expected") == 0)
					arguments->show = LBM_SHOW_EXPECTED;
				else if (strcmp(optarg, "both") == 0)
					arguments->show = LBM_SHOW_BOTH;
				else
					fatal("Invalid value for -s/--show option !");
				break;
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
	lbm_mesh_t * mesh_rank;
	lbm_comm_t comm;
	int rank;
	int comm_size;
	int i;

	//init MPI and get current rank and commuincator size.
	MPI_Init( &argc, &argv );
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	MPI_Comm_size( MPI_COMM_WORLD, &comm_size );

	//parse args
	struct arguments arguments = {
		.exercice = 0,
		.width = 16,
		.height = 16,
		.show = LBM_SHOW_CURRENT,
		.fill = LBM_FILL_MODULO_9,
	};
	parse_prgm_arguments(&arguments, argc, argv);

	//set exo
	lbm_ex_select(arguments.exercice);

	//run only on 9 ranks
	if ( comm_size > 64 ) {
		if ( rank == RANK_MASTER )
			fprintf(stderr, "Can run only with 64 ranks max !\n");
		MPI_Finalize();
		return EXIT_FAILURE;
	}

	//allocate
	mesh_rank = malloc(sizeof(lbm_mesh_t) * comm_size);

	//setup
	MESH_WIDTH = arguments.width;
	MESH_HEIGHT = arguments.height;

	//init mesh and comms
	if ( rank == RANK_MASTER )
		printf(" * Init...\n");
	lbm_comm_init_ex_select( &comm, MESH_WIDTH, MESH_HEIGHT);

	//some checks
	bool error = false;
	if (comm.width == MESH_WIDTH / comm.nb_x) {
		if (rank == 0) printf(RED "You forgot the add the ghost cells while computing comm->width !" RESET "\n");
		error = true;
	}
	if (comm.height == MESH_HEIGHT / comm.nb_y) {
		if (rank == 0) printf(RED "You forgot the add the ghost cells while computing comm->height !" RESET "\n");
		error = true;
	}
	if (error) {
		if (rank == 0) printf(RED ">>>  ERROR  <<<" RESET "\n");
		exit(1);
	}

	//init mesh
	lbm_mesh_init( &mesh, lbm_comm_width( &comm ), lbm_comm_height( &comm ) );

	//init
	if (arguments.fill == LBM_FILL_RANK)
		mesh_init_rank(&mesh, rank);
	else if (arguments.fill == LBM_FILL_MODULO_9)
		mesh_init_mod(&mesh, &comm, 9);
	else if (arguments.fill == LBM_FILL_MODULO_10)
		mesh_init_mod(&mesh, &comm, 10);
	else if (arguments.fill == LBM_FILL_POSITION)
		mesh_init_pos(&mesh, &comm);

	//splitting
	if ( rank == RANK_MASTER )
		printf(" * Splitting: (%d x %d)\n", comm.nb_x, comm.nb_y);
	MPI_Barrier(MPI_COMM_WORLD);

	//get positions
	int coords[64][2];
	MPI_Status status;
	coords[0][0] = comm.rank_x;
	coords[0][1] = comm.rank_y;
	if (rank == RANK_MASTER) {
		for (i = 1 ; i < comm_size ; i++)
			MPI_Recv( coords[i], 2, MPI_INT, i, 0, MPI_COMM_WORLD, &status );
		for (i = 0 ; i < comm_size ; i++)
			printf(" * Rank %d: (%d, %d)\n", i, coords[i][0], coords[i][1]);
	} else {
		MPI_Send( coords[0], 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	//comm
	if (rank == RANK_MASTER)
		printf(" * Communicate...\n");
	lbm_comm_ghost_exchange_ex_select( &comm, &mesh );
	MPI_Barrier(MPI_COMM_WORLD);

	//allocate recieve meshes
	for (i = 0 ; i < comm_size ; i++)
		lbm_mesh_init( &mesh_rank[i], lbm_comm_width( &comm ), lbm_comm_height( &comm ) );
	
	//fetch on rank 0
	if ( rank == RANK_MASTER ) {
		printf(" * fetch...\n");
		memcpy(mesh_rank[0].cells, mesh.cells, mesh.width * mesh.height * DIRECTIONS * sizeof(double));
		for (i = 1 ; i < comm_size ; i++)
			MPI_Recv( mesh_rank[i].cells, mesh_rank[i].width * mesh_rank[i].height * DIRECTIONS, MPI_DOUBLE, i, i, MPI_COMM_WORLD, &status );
		printf(" * display...\n");
		display_meshes(&comm, mesh_rank, coords, comm_size, arguments.show, arguments.fill);
	} else {
		MPI_Send( mesh.cells, mesh.width * mesh.height * DIRECTIONS, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD );
	}

	//clean
	lbm_comm_release_ex_select(&comm);
	lbm_mesh_release(&mesh);
	for (i = 0 ; i < comm_size ; i++)
		lbm_mesh_release(&mesh_rank[i]);
	free(mesh_rank);

	//close MPI
	MPI_Finalize();

	return EXIT_SUCCESS;
}
