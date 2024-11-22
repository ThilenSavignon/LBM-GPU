/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

/****************************************************/
#include <stdlib.h>
#include <mpi.h>
#include "lbm_struct.h"

/****************************************************/
/**
 * Function used to initialize the local mesh.
 * @param mesh Mesh to init.
 * @param width With of the local mesh accounting the ghost cells.
 * @param height Height of the local mesh accounting the ghost cells.
**/
void lbm_mesh_init( lbm_mesh_t * mesh, int width,  int height )
{
	//setup params
	mesh->width = width;
	mesh->height = height;

	//alloc cells memory
	mesh->cells = malloc( width * height  * DIRECTIONS * sizeof( double ) );

	//errors
	if( mesh->cells == NULL )
	{
		perror( "malloc" );
		abort();
	}
}

/****************************************************/
/** Release the memory allocated to store data of the local mesh. **/
void lbm_mesh_release( lbm_mesh_t * mesh )
{
	//reset values
	mesh->width = 0;
	mesh->height = 0;

	//free memory
	free( mesh->cells );
	mesh->cells = NULL;
}

/****************************************************/
/**
 * Function used to initiliazs the cell type local mesh.
 * @param mesh Cell type mesh to initilize.
 * @param width Width of the local mesh, accounting the ghost cells.
 * @param height Height of the local mesh, accounting the ghost cells.
**/
void lbm_mesh_type_t_init( lbm_mesh_type_t * meshtype, int width,  int height )
{
	//setup params
	meshtype->width = width;
	meshtype->height = height;

	//alloc cells memory
	meshtype->types = malloc( (width + 2) * height * sizeof( lbm_cell_type_t ) );

	//errors
	if( meshtype->types == NULL )
	{
		perror( "malloc" );
		abort();
	}
}

/****************************************************/
/** Release the memory allocated to store the local cells type. **/
void lbm_mesh_type_t_release( lbm_mesh_type_t * mesh )
{
	//reset values
	mesh->width = 0;
	mesh->height = 0;

	//free memory
	free( mesh->types );
	mesh->types = NULL;
}

/****************************************************/
/**
 * Function which can be used to abort on error.
 * @param message to be printed.
**/
void fatal(const char * message)
{
	#ifdef DISABLE_COLORS
		fprintf(stderr,"FATAL ERROR : %s\n",message);
	#else
		fprintf(stderr,"\033[31mFATAL ERROR : %s\033[39m\n",message);
	#endif
	MPI_Abort(MPI_COMM_WORLD, -1);
	//MPI_Finalize();
	//exit(1);
}

/****************************************************/
/**
 * Function which can be used to print warnings
 * @param message to be printed.
**/
void warning(const char * message)
{
	#ifdef DISABLE_COLORS
		fprintf(stderr,"Warning : %s\n",message);
	#else
		fprintf(stderr,"\033[33mWarning : %s\033[39m\n",message);
	#endif
	//MPI_Finalize();
	//exit(1);
}
