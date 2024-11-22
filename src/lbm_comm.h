/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

#ifndef comm_H
#define comm_H

/****************************************************/
#include <mpi.h>
#include <stdlib.h>
#include "lbm_struct.h"

/****************************************************/
/** Define the rank to be used as master. **/
#define RANK_MASTER 0
/** Maximum number of parallel async operations to track. **/
#define MAX_ASYNC 16

/****************************************************/
/**
 * Structure used to keep track of the communication settings and
 * local domain splitting.
**/
typedef struct lbm_comm_t_s
{
	/** Number of processes along X. **/
	int nb_x;
	/** Number of processes along Y. **/
	int nb_y;
	/** Process rank position along the X axis. **/
	int rank_x;
	/** Process rank position along the Y axis. **/
	int rank_y;
	/** 
	 * Absolute position along X of the local mesh in the global one 
	 * without accounting the ghost cells. 
	**/
	int x;
	/** 
	 * Absolute position along Y of the local mesh in the global one 
	 * without accounting the ghost cells. 
	**/
	int y;
	/** Width of the local mesh, accounting the ghost cells. **/
	int width;
	/** Height of the local mesh, accounting the ghost cells. **/
	int height;
	/** Can be used to store the cartesian communication if using MPI_Cart. **/
	MPI_Comm communicator;
	/** Can be used to store requests. **/
	MPI_Request requests[MAX_ASYNC];
	/** Can be used to store data type. **/
	MPI_Datatype type;
	MPI_Datatype type2;
	MPI_Datatype type3;
	/** Can be used to keep track of buffer for non contiguous communications. **/
	double * buffer_send_up;
	/** Can be used to keep track of buffer for non contiguous communications. **/
	double * buffer_send_down;
	/** Can be used to keep track of buffer for non contiguous communications. **/
	double * buffer_recv_up;
	/** Can be used to keep track of buffer for non contiguous communications. **/
	double * buffer_recv_down;
	//////////////////// EXTRA PARAMETERS ALREADY HANDLED /////////////////////////
	/** Keep track of the file handler to save data (DO NOT MODIFY FOR THE LAB) **/
	MPI_File file_handler;
} lbm_comm_t;

/****************************************************/
static inline int lbm_comm_width( lbm_comm_t *comm )
{
	return comm->width;
}

/****************************************************/
static inline int lbm_comm_height( lbm_comm_t *comm )
{
	return comm->height;
}

/****************************************************/
void  lbm_comm_print( lbm_comm_t * comm );

#endif
