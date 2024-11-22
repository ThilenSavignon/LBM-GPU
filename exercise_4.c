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
//       8 neighbors using manual copy for non
//       contiguous side and blocking communications
//
// SUMMARY:
//     - 2D splitting along X and Y
//     - 8 neighbors communications
//     - Blocking communications
//     - Manual copy for non continguous cells
//
//////////////////////////////////////////////////////

/****************************************************/
#include "mpi.h"
#include "src/lbm_struct.h"
#include "src/exercises.h"
#include <assert.h>
#include <string.h>

/****************************************************/
void lbm_comm_init_ex4(lbm_comm_t *comm, int total_width, int total_height) {
    // DONE: calculate the splitting parameters for the current task.
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size; MPI_Comm_size(MPI_COMM_WORLD, &size);
    int dims[2] = {0, 0}; assert(MPI_Dims_create(size, 2, dims) == MPI_SUCCESS);

    assert(total_width % dims[0] == 0);
    assert(total_height % dims[1] == 0);
    // DONE: calculate the number of tasks along X axis and Y axis.
    comm->nb_x = dims[0];
    comm->nb_y = dims[1];

    int periods[2] = {0, 0};
    assert(MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm->communicator) == MPI_SUCCESS);
    int coords[2]; assert(MPI_Cart_coords(comm->communicator, rank, 2, coords) == MPI_SUCCESS);

    // DONE: calculate the current task position in the splitting
    comm->rank_x = coords[0];
    comm->rank_y = coords[1];

    // DONE: calculate the local sub-domain size (do not forget the 
    //       ghost cells). Use total_width & total_height as starting 
    //       point.
    comm->width = total_width / comm->nb_x + 2;
    comm->height = total_height / comm->nb_y + 2;

    // DONE: calculate the absolute position  (in cell number) in the global mesh.
    //       without accounting the ghost cells
    //       (used to setup the obstable & initial conditions).
    comm->x = comm->rank_x * (comm->width - 2);
    comm->y = comm->rank_y * (comm->height - 2);

    // OPTIONAL: if you want to avoid allocating temporary copy buffer
    //           for every step :
    comm->buffer_recv_down = (double *) malloc(sizeof(double) * DIRECTIONS * comm->width);
    comm->buffer_recv_up = (double *) malloc(sizeof(double) * DIRECTIONS * comm->width);
    comm->buffer_send_down = (double *) malloc(sizeof(double) * DIRECTIONS * comm->width);
    comm->buffer_send_up = (double *) malloc(sizeof(double) * DIRECTIONS * comm->width);

    // if debug print comm
    lbm_comm_print(comm);
}

/****************************************************/
void lbm_comm_release_ex4(lbm_comm_t *comm) {
    // free allocated ressources
    free(comm->buffer_recv_down);
    free(comm->buffer_recv_up);
    free(comm->buffer_send_down);
    free(comm->buffer_send_up);
}

/****************************************************/
void lbm_comm_ghost_exchange_ex4(lbm_comm_t *comm, lbm_mesh_t *mesh) {
    // 
    // DONE: Implement the 2D communication with :
    //         - blocking MPI functions
    //         - manual copy in temp buffer for non contiguous side 
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
    // 
    // example to access cell
    // double * cell = lbm_mesh_get_cell(mesh, local_x, local_y);
    // double * cell = lbm_mesh_get_cell(mesh, comm->width - 1, 0);
    // 
    // DONE:
    //    - implement left/write communications
    //    - implement top/bottom communication (non contiguous)
    //    - implement diagonal communications

    int leftRank, rightRank; MPI_Cart_shift(comm->communicator, 0, 1, &leftRank, &rightRank);
    int  topRank,   botRank; MPI_Cart_shift(comm->communicator, 1, 1,  &topRank,   &botRank);

    int trueWidth = comm->width * DIRECTIONS;
    int trueHeight = comm->height * DIRECTIONS;
    int cellSize = DIRECTIONS * sizeof(double);

    MPI_Status status;
    if ((comm->rank_x + comm->rank_y) % 2 == 0) {
        if (leftRank != MPI_PROC_NULL) {
            MPI_Recv(lbm_mesh_get_cell(mesh,               0, 0), trueHeight, MPI_DOUBLE, leftRank, 0, comm->communicator, &status);  // Receive left column
            MPI_Send(lbm_mesh_get_cell(mesh,               1, 0), trueHeight, MPI_DOUBLE, leftRank, 0, comm->communicator);  // Send left column
        }
        if (rightRank != MPI_PROC_NULL) {
            MPI_Recv(lbm_mesh_get_cell(mesh, comm->width - 1, 0), trueHeight, MPI_DOUBLE, rightRank, 0, comm->communicator, &status);  // Receive right column
            MPI_Send(lbm_mesh_get_cell(mesh, comm->width - 2, 0), trueHeight, MPI_DOUBLE, rightRank, 0, comm->communicator);  // Send right column
        }
        if (topRank != MPI_PROC_NULL) {
            MPI_Recv(comm->buffer_recv_up                       ,  trueWidth, MPI_DOUBLE,   topRank, 0, comm->communicator, &status);  // Receive top row
            for (int i = 0; i < comm->width; i++) {
                memcpy(lbm_mesh_get_cell(mesh, i,                0), &comm->buffer_recv_up[i * 9]                , cellSize);
                memcpy(&comm->buffer_send_up[i * 9]                , lbm_mesh_get_cell(mesh, i,                1), cellSize);
            }
            MPI_Send(comm->buffer_send_up                       ,  trueWidth, MPI_DOUBLE,   topRank, 0, comm->communicator);  // Send top row
        }
        if (botRank != MPI_PROC_NULL) {
            MPI_Recv(comm->buffer_recv_down                     ,  trueWidth, MPI_DOUBLE,   botRank, 0, comm->communicator, &status);  // Receive bot row
            for (int i = 0; i < comm->width; i++) {
                memcpy(lbm_mesh_get_cell(mesh, i, comm->height - 1), &comm->buffer_recv_down[i * 9]              , cellSize);
                memcpy(&comm->buffer_send_down[i * 9]              , lbm_mesh_get_cell(mesh, i, comm->height - 2), cellSize);
            }
            MPI_Send(comm->buffer_send_down                     ,  trueWidth, MPI_DOUBLE,   botRank, 0, comm->communicator);  // Send bot row
        }
    } else {
        if (leftRank != MPI_PROC_NULL) {
            MPI_Send(lbm_mesh_get_cell(mesh,               1, 0), trueHeight, MPI_DOUBLE,  leftRank, 0, comm->communicator);  // Send left column
            MPI_Recv(lbm_mesh_get_cell(mesh,               0, 0), trueHeight, MPI_DOUBLE,  leftRank, 0, comm->communicator, &status);  // Receive left column
        }
        if (rightRank != MPI_PROC_NULL) {
            MPI_Send(lbm_mesh_get_cell(mesh, comm->width - 2, 0), trueHeight, MPI_DOUBLE, rightRank, 0, comm->communicator);  // Send right column
            MPI_Recv(lbm_mesh_get_cell(mesh, comm->width - 1, 0), trueHeight, MPI_DOUBLE, rightRank, 0, comm->communicator, &status);  // Receive right column
        }
        if (topRank != MPI_PROC_NULL) {
            for (int i = 0; i < comm->width; i++)
                memcpy(&comm->buffer_send_up[i * 9]                , lbm_mesh_get_cell(mesh, i,                1), cellSize);
            MPI_Send(comm->buffer_send_up                       ,  trueWidth, MPI_DOUBLE,   topRank, 0, comm->communicator);  // Send top row
            MPI_Recv(comm->buffer_recv_up                       ,  trueWidth, MPI_DOUBLE,   topRank, 0, comm->communicator, &status);  // Receive top row
            for (int i = 0; i < comm->width; i++)
                memcpy(lbm_mesh_get_cell(mesh, i, 0)               , &comm->buffer_recv_up[i * 9]                , cellSize);
            
        }
        if (botRank != MPI_PROC_NULL) {
            for (int i = 0; i < comm->width; i++)
                memcpy(&comm->buffer_send_down[i * 9]              , lbm_mesh_get_cell(mesh, i, comm->height - 2), cellSize);
            MPI_Send(comm->buffer_send_down                     ,  trueWidth, MPI_DOUBLE,   botRank, 0, comm->communicator);  // Send bot row
            MPI_Recv(comm->buffer_recv_down                     ,  trueWidth, MPI_DOUBLE,   botRank, 0, comm->communicator, &status);  // Receive bot row
            for (int i = 0; i < comm->width; i++)
                memcpy(lbm_mesh_get_cell(mesh, i, comm->height - 1), &comm->buffer_recv_down[i * 9]              , cellSize);           
        }
    }
}
