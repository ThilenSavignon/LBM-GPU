/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

/****************************************************/
#ifndef LBM_SAVE_H
#define LBM_SAVE_H

/****************************************************/
#include <stdio.h>
#include <mpi.h>
#include "lbm_comm.h"
#include "lbm_struct.h"
#include "lbm_save.h"

/****************************************************/
typedef struct lbm_file_mesh_s {
	lbm_file_entry_t * cells;
	int width;
	int height;
} lbm_file_mesh_t;

/****************************************************/
void lbm_save_mesh_init(lbm_file_mesh_t * file_mesh, lbm_comm_t * comm);
void lbm_save_mesh_release(lbm_file_mesh_t * file_mesh);
void lbm_save_fill_mesh(lbm_file_mesh_t * file_mesh, const lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type);
void lbm_save_write_mesh(lbm_file_mesh_t * file_mesh, lbm_comm_t * comm, int rank_x, int rank_y, int write_step);

/****************************************************/
void lbm_save_file_header(lbm_comm_t * comm);
void lbm_open_output_file(lbm_comm_t * comm);

#endif //LBM_SAVE_H
