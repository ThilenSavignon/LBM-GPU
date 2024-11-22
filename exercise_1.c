/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
//
// GOAL: Implement a 1D communication scheme along
//       X axis with blocking communications.
//
// SUMMARY:
//     - 1D splitting along X
//     - Blocking communications
//
//////////////////////////////////////////////////////

/****************************************************/
#include "mpi.h"
#include "assert.h"
#include "src/lbm_config.h"
#include "src/lbm_struct.h"
#include "src/exercises.h"

/****************************************************/
void lbm_comm_init_ex1(lbm_comm_t *comm, int total_width, int total_height) {
    //
    // DONE: calculate the splitting parameters for the current task.
    //
    // HINT: You can look in exercise_0.c to get an example for the sequential case.
    int size; MPI_Comm_size(MPI_COMM_WORLD, &size);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // DONE: calculate the number of tasks along X axis and Y axis.
    assert(total_width % size == 0);
    comm->nb_x = size;
    comm->nb_y = 1;

    // DONE: calculate the current task position in the splitting
    comm->rank_x = rank;
    comm->rank_y = 0;

    // DONE: calculate the local sub-domain size (do not forget the 
    //       ghost cells). Use total_width & total_height as starting 
    //       point.
    comm->width = (total_width / size) + 2;
    comm->height = total_height + 2;

    // DONE: calculate the absolute position in the global mesh.
    //       without accounting the ghost cells
    //       (used to setup the obstable & initial conditions).
    comm->x = rank * (comm->width - 2);
    comm->y = 0;

    //if debug print comm
    lbm_comm_print(comm);
}

/****************************************************/
void lbm_comm_ghost_exchange_ex1(lbm_comm_t *comm, lbm_mesh_t *mesh) {
    //
    // DONE: Implement the 1D communication with blocking MPI functions (MPI_Send & MPI_Recv)
    //
    // To be used:
    //    - DIRECTIONS: the number of doubles composing a cell
    //    - double[DIRECTIONS] lbm_mesh_get_cell(mesh, x, y): function to get the address of a particular cell.
    //    - comm->width : The with of the local sub-domain (containing the ghost cells)
    //    - comm->height : The height of the local sub-domain (containing the ghost cells)
	
    // example to access cell
    // double * cell = lbm_mesh_get_cell(mesh, local_x, local_y);
    // double * cell = lbm_mesh_get_cell(mesh, comm->width - 1, 0);
    
    // Sending the columns
    
    MPI_Status status;
    if (comm->rank_x > 0)
        MPI_Recv(lbm_mesh_get_cell(mesh, 0, 0), comm->height * DIRECTIONS, MPI_DOUBLE, comm->rank_x - 1, 0, MPI_COMM_WORLD, &status);
    if (comm->rank_x < comm->nb_x - 1) {
        MPI_Send(lbm_mesh_get_cell(mesh, comm->width - 2, 0), comm->height * DIRECTIONS, MPI_DOUBLE, comm->rank_x + 1, 0, MPI_COMM_WORLD);
        MPI_Recv(lbm_mesh_get_cell(mesh, comm->width - 1, 0), comm->height * DIRECTIONS, MPI_DOUBLE, comm->rank_x + 1, 0, MPI_COMM_WORLD, &status);
    }
    if (comm->rank_x > 0)
        MPI_Send(lbm_mesh_get_cell(mesh, 1, 0), comm->height * DIRECTIONS, MPI_DOUBLE, comm->rank_x - 1, 0, MPI_COMM_WORLD);
}

