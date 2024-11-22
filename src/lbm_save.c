/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

/****************************************************/
#include <mpi.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include "lbm_phys.h"
#include "lbm_save.h"

/****************************************************/
/**
 * Save the header of the output file. This header mainly contains the informations about
 * the mesh to know how to load it and more from frame to frame.
 * @param comm Communication structure to keep track of the MPI_File handler.
**/
void lbm_save_file_header(lbm_comm_t * comm)
{
	//setup header values
	lbm_file_header_t header;
	header.magick      = RESULT_MAGICK;
	header.mesh_height = MESH_HEIGHT;
	header.mesh_width  = MESH_WIDTH;
	header.lines       = comm->nb_y;

	//write file
	//fwrite(&header,sizeof(header),1,fp);
	//ftruncate(fileno(fp), 0);
	int status = MPI_File_write_at(comm->file_handler, 0, &header, sizeof(header), MPI_CHAR, MPI_STATUS_IGNORE);
	if (status != MPI_SUCCESS)
		fatal("Failed to write header in output file !");
}

/****************************************************/
/**
 * Function to be used to get a cell from its coordinate.
 * @param mesh Pointer to the mesh struct containing the data loaded from the file.
 * @param x Coordinate of the cell in the local mesh withoud accounting the ghost cells.
 * @param y coordinate of the cell in the local mesh withoud accounting the ghost cells.
**/
static inline lbm_file_entry_t * lbm_file_mesh_get_cell( const lbm_file_mesh_t * mesh, int x, int y)
{
	return &mesh->cells[ (x * mesh->height + y) ];
}

/****************************************************/
void lbm_save_mesh_init(lbm_file_mesh_t * file_mesh, lbm_comm_t * comm)
{
	//check
	assert(file_mesh != NULL);
	assert(comm != NULL);

	//size
	file_mesh->width = comm->width - 2;
	file_mesh->height = comm->height - 2;

	//allocate
	file_mesh->cells = malloc( sizeof(lbm_file_entry_t) * file_mesh->width * file_mesh->height );
}

/****************************************************/
void lbm_save_mesh_release(lbm_file_mesh_t * file_mesh)
{
	//check
	assert(file_mesh != NULL);

	//free
	free(file_mesh->cells);
}

/****************************************************/
void lbm_save_fill_mesh(lbm_file_mesh_t * file_mesh, const lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type)
{
	//write buffer to write float instead of double
	int i,j;
	double density;
	Vector v;
	double norm;

	//nothing to do
	if (RESULT_FILENAME == NULL)
		return;

	//loop on all values
	for ( i = 1 ; i < mesh->width - 1 ; i++)
	{
		for ( j = 1 ; j < mesh->height - 1 ; j++)
		{
			//compute macrospic values
			density = lbm_phys_cell_density(lbm_mesh_get_cell(mesh, i, j));
			lbm_phys_cell_velocity(v,lbm_mesh_get_cell(mesh, i, j),density);
			norm = sqrt(lbm_phys_vect_norme_2(v,v));

			//fill obstable
			if (*lbm_cell_type_t_get_cell(mesh_type, i, j) == CELL_BOUNCE_BACK) {
				norm = NAN;
				density = NAN;
			}

			//fill
			lbm_file_entry_t * cell = lbm_file_mesh_get_cell(file_mesh, i - 1, j - 1);
			cell->density = density;
			cell->v = norm;
		}
	}
}

/****************************************************/
void lbm_save_write_mesh(lbm_file_mesh_t * file_mesh, lbm_comm_t * comm, int rank_x, int rank_y, int write_step)
{
	//checks
	assert(file_mesh != NULL);
	assert(comm != NULL);
	assert(rank_x >= 0 && rank_x < comm->nb_x);
	assert(rank_y >= 0 && rank_y < comm->nb_y);

	//nothing to write
	if (RESULT_FILENAME == NULL)
		return;

	//calc size
	size_t size = sizeof(lbm_file_entry_t) * file_mesh->width * file_mesh->height;

	//calc offset
	size_t offset = sizeof(lbm_file_header_t) + (comm->nb_x * comm->nb_y * size * write_step) + (rank_y * comm->nb_x + rank_x) * size;

	//pwrite
	int status = MPI_File_write_at(comm->file_handler, offset, file_mesh->cells, size, MPI_CHAR, MPI_STATUS_IGNORE);
	if (status != MPI_SUCCESS)
		fatal("Fail to fully write data into file !");
}

/****************************************************/
void lbm_open_output_file(lbm_comm_t * comm)
{
	//check if empty filename => so noout
	if (RESULT_FILENAME == NULL)
		return;

	//get infos
	int rank;
	MPI_Comm_rank( MPI_COMM_WORLD, &rank );

	//open result file
	int status = MPI_File_open(MPI_COMM_WORLD, RESULT_FILENAME, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &comm->file_handler);

	//errors
	if (status != MPI_SUCCESS)
	{
		fprintf(stderr, "Fail to open file %s\n", RESULT_FILENAME);
		abort();
	}

	//write header
	if (rank == 0) {
		printf("write header \n");
		lbm_save_file_header(comm);
	}
}
