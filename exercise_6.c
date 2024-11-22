/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

//////////////////////////////////////////////////////
//
// Goal: Implement 2D grid communication with non-blocking
//       messages.
//
// SUMMARY:
//     - 2D splitting along X and Y
//     - 8 neighbors communications
//     - MPI type for non contiguous cells
// NEW:
//     - Non-blocking communications
//
//////////////////////////////////////////////////////

/****************************************************/
#include "src/lbm_struct.h"
#include <assert.h>
#include <string.h>
#include "mpi.h"
#include "src/lbm_config.h"
#include "src/exercises.h"

void lbm_comm_init(lbm_comm_t *comm, int total_width, int total_height){
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
void lbm_comm_init_ex6(lbm_comm_t *comm, int total_width, int total_height) {
    // we use the same implementation than ex5 execpt for type creation
    lbm_comm_init(comm, total_width, total_height);

    MPI_Type_vector(
        comm->width - 2,
        DIRECTIONS,
        comm->height * DIRECTIONS,
        MPI_DOUBLE,
        &comm->type
    );
    MPI_Type_commit(&comm->type);

    MPI_Type_vector(
        comm->width - 1,
        DIRECTIONS,
        comm->height * DIRECTIONS,
        MPI_DOUBLE,
        &comm->type2
    );
    MPI_Type_commit(&comm->type2);
    
    MPI_Type_vector(
        comm->width,
        DIRECTIONS,
        comm->height * DIRECTIONS,
        MPI_DOUBLE,
        &comm->type3
    );
    MPI_Type_commit(&comm->type3);
}

/****************************************************/
void lbm_comm_release_ex6(lbm_comm_t *comm) {
    // we use the same implementation than ext 5
    // free allocated ressources
    free(comm->buffer_recv_down);
    free(comm->buffer_recv_up);
    free(comm->buffer_send_down);
    free(comm->buffer_send_up);

    // DONE: release MPI type created in init.
    MPI_Type_free(&comm->type);
    MPI_Type_free(&comm->type2);
    MPI_Type_free(&comm->type3);
}

/****************************************************/
void lbm_comm_ghost_exchange_ex6(lbm_comm_t *comm, lbm_mesh_t *mesh) {
    //
    // DONE: Implement the 2D communication with :
    //    - non-blocking MPI functions
    //    - use MPI type for non contiguous side 
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
    // TIP: The previous trick require to make two batch of non-blocking communications.

    // example to access cell
    // double * cell = lbm_mesh_get_cell(mesh, local_x, local_y);
    // double * cell = lbm_mesh_get_cell(mesh, comm->width - 1, 0);

    // DONE:
    //    - implement left/write communications
    //    - implement top/bottom communication (non contiguous)
    //    - implement diagonal communications
    
    int leftRank, rightRank; MPI_Cart_shift(comm->communicator, 0, 1, &leftRank, &rightRank);
    int topRank, botRank; MPI_Cart_shift(comm->communicator, 1, 1, &topRank, &botRank);
    
    int topLeftRank, topRightRank, botLeftRank, botRightRank;
    topLeftRank = topRightRank = botLeftRank = botRightRank = MPI_PROC_NULL;
    int coords[2] = {comm->rank_x, comm->rank_y - 1};
    if (comm->rank_y > 0) {
        if (comm->rank_x > 0) {
            coords[0]--; MPI_Cart_rank(comm->communicator, coords, &topLeftRank); coords[0]++;
        }
        if (comm->rank_x < comm->nb_x - 1) {
            coords[0]++; MPI_Cart_rank(comm->communicator, coords, &topRightRank); coords[0]--;
        }
    }
    coords[1] += 2;
    if (comm->rank_y < comm->nb_y - 1) {
        if (comm->rank_x > 0) {
            coords[0]--; MPI_Cart_rank(comm->communicator, coords, &botLeftRank); coords[0]++;
        }
        if (comm->rank_x < comm->nb_x - 1) {
            coords[0]++; MPI_Cart_rank(comm->communicator, coords, &botRightRank);
        }
    }
    
    int rcount = 0;
    MPI_Request requests[16];

    int from, to;
    MPI_Datatype type;

    // Borders (including ghost corners)
    from = (topRank != MPI_PROC_NULL);
    to = comm->height - from - (botRank != MPI_PROC_NULL);
    if (leftRank != MPI_PROC_NULL) {
        MPI_Irecv(lbm_mesh_get_cell(mesh,               0,             from), to * DIRECTIONS, MPI_DOUBLE,     leftRank, 0, comm->communicator, &requests[rcount++]);
        MPI_Isend(lbm_mesh_get_cell(mesh,               1,             from), to * DIRECTIONS, MPI_DOUBLE,     leftRank, 0, comm->communicator, &requests[rcount++]);
    }
    if (rightRank != MPI_PROC_NULL) {
        MPI_Irecv(lbm_mesh_get_cell(mesh, comm->width - 1,             from), to * DIRECTIONS, MPI_DOUBLE,    rightRank, 0, comm->communicator, &requests[rcount++]);
        MPI_Isend(lbm_mesh_get_cell(mesh, comm->width - 2,             from), to * DIRECTIONS, MPI_DOUBLE,    rightRank, 0, comm->communicator, &requests[rcount++]);
    }
    from = (leftRank != MPI_PROC_NULL);
    type = (from != (rightRank != MPI_PROC_NULL)) ? comm->type2 : (from == 0 ? comm->type3 : comm->type);
    if (topRank != MPI_PROC_NULL) {
        MPI_Irecv(lbm_mesh_get_cell(mesh,            from,                0),               1,       type,      topRank, 0, comm->communicator, &requests[rcount++]);
        MPI_Isend(lbm_mesh_get_cell(mesh,            from,                1),               1,       type,      topRank, 0, comm->communicator, &requests[rcount++]);
    }
    if (botRank != MPI_PROC_NULL) {
        MPI_Irecv(lbm_mesh_get_cell(mesh,            from, comm->height - 1),               1,       type,      botRank, 0, comm->communicator, &requests[rcount++]);
        MPI_Isend(lbm_mesh_get_cell(mesh,            from, comm->height - 2),               1,       type,      botRank, 0, comm->communicator, &requests[rcount++]);
    }

    // Corners
    if (topLeftRank != MPI_PROC_NULL) {
        MPI_Irecv(lbm_mesh_get_cell(mesh,               0,                0),      DIRECTIONS, MPI_DOUBLE,  topLeftRank, 0, comm->communicator, &requests[rcount++]);
        MPI_Isend(lbm_mesh_get_cell(mesh,               1,                1),      DIRECTIONS, MPI_DOUBLE,  topLeftRank, 0, comm->communicator, &requests[rcount++]);
    }
    if (topRightRank != MPI_PROC_NULL) {
        MPI_Irecv(lbm_mesh_get_cell(mesh, comm->width - 1,                0),      DIRECTIONS, MPI_DOUBLE, topRightRank, 0, comm->communicator, &requests[rcount++]);
        MPI_Isend(lbm_mesh_get_cell(mesh, comm->width - 2,                1),      DIRECTIONS, MPI_DOUBLE, topRightRank, 0, comm->communicator, &requests[rcount++]);
    }
    if (botLeftRank != MPI_PROC_NULL) {
        MPI_Irecv(lbm_mesh_get_cell(mesh,               0, comm->height - 1),      DIRECTIONS, MPI_DOUBLE,  botLeftRank, 0, comm->communicator, &requests[rcount++]);
        MPI_Isend(lbm_mesh_get_cell(mesh,               1, comm->height - 2),      DIRECTIONS, MPI_DOUBLE,  botLeftRank, 0, comm->communicator, &requests[rcount++]);
    }
    if (botRightRank != MPI_PROC_NULL) {
        MPI_Irecv(lbm_mesh_get_cell(mesh, comm->width - 1, comm->height - 1),      DIRECTIONS, MPI_DOUBLE, botRightRank, 0, comm->communicator, &requests[rcount++]);
        MPI_Isend(lbm_mesh_get_cell(mesh, comm->width - 2, comm->height - 2),      DIRECTIONS, MPI_DOUBLE, botRightRank, 0, comm->communicator, &requests[rcount++]);
    }

    MPI_Waitall(rcount, requests, MPI_STATUSES_IGNORE);
}

