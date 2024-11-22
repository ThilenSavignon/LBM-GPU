/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

#ifndef LBM_STRUCT_H
#define LBM_STRUCT_H

/****************************************************/
#include <stdint.h>
#include <stdio.h>
#include "lbm_config.h"

/****************************************************/
/**
 * A cell is an array of DIRECTIONS doubles to store the microscopic
 * probabilities (f_i)
**/
typedef double * lbm_mesh_cell_t;
/** Represent a vector to handle the macroscopic verlocity. **/
typedef double Vector[DIMENSIONS];

/****************************************************/
/**
 * Define a mesh to represent the data of the local domain. This mesh also 
 * contain the ghost cells arround the local cells.
**/
typedef struct lbm_mesh_s
{
	/** Cells of the mesh (MESH_WIDTH * MESG_HEIGHT). **/
	double * cells;
	/** Width of the local mesh (accounting the ghost cells). **/
	int width;
	/** Height of the local mesh (accounting the ghost cells). **/
	int height;
} lbm_mesh_t;

/****************************************************/
/**
 * Define the type of the cell to know what compute function to apply.
**/
typedef enum lbm_cell_type_e
{
	/** Standard fluid cells on which to apply only the lbm_phys_collisions function. **/
	CELL_FUILD,
	/** Cells of the obstable of top and bottom border. Apply reflection. **/
	CELL_BOUNCE_BACK,
	/** Cells of the input wall. Apply the Zou/He with a fixed V. **/
	CELL_LEFT_IN,
	/** Cells of the output wall. Apply Zou/He with gradiant and constant density. **/
	CELL_RIGHT_OUT
} lbm_cell_type_t;

/****************************************************/
/**
 * Matrix storing the type of each cell of the mesh (accounting ghost cells).
**/
typedef struct lbm_mesh_type_s
{
	/** Store the type of the local cells (MESH_WIDTH * MESH_HEIGHT). **/
	lbm_cell_type_t * types;
	/** width of the local type mesh (mailles fantome comprises). **/
	int width;
	/** Height of the local type mesh (mailles fantome comprises). **/
	int height;
} lbm_mesh_type_t;

/****************************************************/
/** Structure used as header for the output file. **/
typedef struct lbm_file_header_s
{
	/** Magick number to check the type of file. **/
	uint32_t magick;
	/** Width of the global mesh (no ghost cells). **/
	uint32_t mesh_width;
	/** Height of the global mesh (no ghost cells). **/
	uint32_t mesh_height;
	/** Number of vertical lines. **/
	uint32_t lines;
} lbm_file_header_t;

/****************************************************/
/** Representation of a cell in the output file with two macroscopic numbers. **/
typedef struct lbm_file_entry_s
{
	/** Velocity of the fluid. **/
	float v;
	/** Density of the fluild. **/
	float density;
} lbm_file_entry_t;

/****************************************************/
/** Use to read the output file. **/
typedef struct lbm_data_file_s
{
	/** Keep track of the file pointer **/
	FILE * fp;
	/** Content of the headers. **/
	lbm_file_header_t header;
	/** Loaded data for the current frame. **/
	lbm_file_entry_t * entries;
} lbm_data_file_t;


/****************************************************/
void lbm_mesh_init( lbm_mesh_t * mesh, int width,  int height );
void lbm_mesh_release( lbm_mesh_t * mesh );

/****************************************************/
void lbm_mesh_type_t_init( lbm_mesh_type_t * mesh, int width,  int height );
void lbm_mesh_type_t_release( lbm_mesh_type_t * mesh );

/****************************************************/
void fatal(const char * message);
void warning(const char * message);

/****************************************************/
/**
 * Function used to get the address of a given cell in the local mesh.
 * @param mesh Pointer to the mesh struct.
 * @param x Position of the cell in the local mesh (accounting ghost cells)
 * @param y Position of the cell in the local mesh (accounting ghost cells)
**/
static inline double * lbm_mesh_get_cell( const lbm_mesh_t * mesh, int x, int y)
{
	return &mesh->cells[ (x * mesh->height + y) * DIRECTIONS ];
}

/****************************************************/
/**
 * Function used to get the address of a given cell type in the local mesh.
 * @param mesh Pointer to the mesh struct.
 * @param x Position of the cell in the local mesh (accounting ghost cells)
 * @param y Position of the cell in the local mesh (accounting ghost cells)
**/
static inline lbm_cell_type_t * lbm_cell_type_t_get_cell( const lbm_mesh_type_t * meshtype, int x, int y)
{
	return &meshtype->types[ x * meshtype->height + y];
}

#endif //LBM_STRUCT_H
