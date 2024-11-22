/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
// Goal: Implement 2D grid communication scheme with
//      8 neighbors using MPI types for non contiguous
//      side.
//
// SUMMARY:
//     - 2D splitting along X and Y
//     - 8 neighbors communications
//     - Blocking communications
// NEW:
//     - >>> MPI type for non contiguous cells <<<
//
//////////////////////////////////////////////////////

/****************************************************/
#include "mpi.h"
#include "src/lbm_config.h"
#include "src/lbm_struct.h"
#include "src/exercises.h"

/****************************************************/
void lbm_comm_init_ex5(lbm_comm_t *comm, int total_width, int total_height)
{
    // we use the same implementation than ex5 execpt for type creation
    lbm_comm_init_ex4(comm, total_width, total_height);

    // DONE: create MPI type for non contiguous side in comm->type.
    MPI_Type_vector(
        comm->width,
        DIRECTIONS,
        comm->height * DIRECTIONS,
        MPI_DOUBLE,
        &comm->type
    );
    MPI_Type_commit(&comm->type);
}

/****************************************************/
void lbm_comm_release_ex5(lbm_comm_t *comm)
{
    // we use the same implementation than ex5 except for type destroy
    lbm_comm_release_ex4(comm);

    // DONE: release MPI type created in init.
    MPI_Type_free(&comm->type);
}

/****************************************************/
void lbm_comm_ghost_exchange_ex5(lbm_comm_t *comm, lbm_mesh_t *mesh) {
    //
    // DONE: Implement the 2D communication with :
    //         - blocking MPI functions
    //         - use MPI type for non contiguous side 
    //
    // To be used:
    //    - DIRECTIONS: the number of doubles composing a cell
    //    - double[9] lbm_mesh_get_cell(mesh, x, y): function to get the address of a particular cell.
    //    - comm->width : The with of the local sub-domain (containing the ghost cells)
    //    - comm->height : The height of the local sub-domain (containing the ghost cells)
    //
    // TIP: create a function to get the target rank from x,y task coordinate.
    // TIP: You can use MPI_PROC_NULL on borders.
    // TIP: send the corner values 2 times, with the up/down/left/write communication
    //      and with the diagonal communication in a second time, this avoid
    //      special cases for border tasks.

    // example to access cell
    // double * cell = lbm_mesh_get_cell(mesh, local_x, local_y);
    // double * cell = lbm_mesh_get_cell(mesh, comm->width - 1, 0);

    // DONE:
    //    - implement left/write communications
    //    - implement top/bottom communication (non contiguous)
    //    - implement diagonal communications

    int leftRank, rightRank; MPI_Cart_shift(comm->communicator, 0, 1, &leftRank, &rightRank);
    int topRank, botRank; MPI_Cart_shift(comm->communicator, 1, 1, &topRank, &botRank);
    
    int trueHeight = comm->height * DIRECTIONS;

    MPI_Status status;
    if ((comm->rank_x + comm->rank_y) % 2 == 0) {
        if (leftRank != MPI_PROC_NULL) {
            MPI_Recv(lbm_mesh_get_cell(mesh,               0,                0), trueHeight, MPI_DOUBLE, leftRank, 0, comm->communicator, &status);  // Receive left column
            MPI_Send(lbm_mesh_get_cell(mesh,               1,                0), trueHeight, MPI_DOUBLE, leftRank, 0, comm->communicator);  // Send left column
        }
        if (rightRank != MPI_PROC_NULL) {
            MPI_Recv(lbm_mesh_get_cell(mesh, comm->width - 1,                0), trueHeight, MPI_DOUBLE, rightRank, 0, comm->communicator, &status);  // Receive right column
            MPI_Send(lbm_mesh_get_cell(mesh, comm->width - 2,                0), trueHeight, MPI_DOUBLE, rightRank, 0, comm->communicator);  // Send right column
        }
        if (topRank != MPI_PROC_NULL) {
            MPI_Recv(lbm_mesh_get_cell(mesh,               0,                0),          1, comm->type,   topRank, 0, comm->communicator, &status);  // Receive top row
            MPI_Send(lbm_mesh_get_cell(mesh,               0,                1),          1, comm->type,   topRank, 0, comm->communicator);  // Send top row
        }
        if (botRank != MPI_PROC_NULL) {
            MPI_Recv(lbm_mesh_get_cell(mesh,               0, comm->height - 1),          1, comm->type,   botRank, 0, comm->communicator, &status);  // Receive bot row
            MPI_Send(lbm_mesh_get_cell(mesh,               0, comm->height - 2),          1, comm->type,   botRank, 0, comm->communicator);  // Send bot row
        }
    } else {
        if (leftRank != MPI_PROC_NULL) {
            MPI_Send(lbm_mesh_get_cell(mesh,               1,                0), trueHeight, MPI_DOUBLE,  leftRank, 0, comm->communicator);  // Send left column
            MPI_Recv(lbm_mesh_get_cell(mesh,               0,                0), trueHeight, MPI_DOUBLE,  leftRank, 0, comm->communicator, &status);  // Receive left column
        }
        if (rightRank != MPI_PROC_NULL) {
            MPI_Send(lbm_mesh_get_cell(mesh, comm->width - 2,                0), trueHeight, MPI_DOUBLE, rightRank, 0, comm->communicator);  // Send right column
            MPI_Recv(lbm_mesh_get_cell(mesh, comm->width - 1,                0), trueHeight, MPI_DOUBLE, rightRank, 0, comm->communicator, &status);  // Receive right column
        }
        if (topRank != MPI_PROC_NULL) {
            MPI_Send(lbm_mesh_get_cell(mesh,               0,                1),          1, comm->type,   topRank, 0, comm->communicator);  // Send top row
            MPI_Recv(lbm_mesh_get_cell(mesh,               0,                0),          1, comm->type,   topRank, 0, comm->communicator, &status);  // Receive top row
        }
        if (botRank != MPI_PROC_NULL) {
            MPI_Send(lbm_mesh_get_cell(mesh,               0, comm->height - 2),          1, comm->type,   botRank, 0, comm->communicator);  // Send bot row
            MPI_Recv(lbm_mesh_get_cell(mesh,               0, comm->height - 1),          1, comm->type,   botRank, 0, comm->communicator, &status);  // Receive bot row
        }
    }
}

