/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

/****************************************************/
#ifndef LBM_INIT_H
#define LBM_INIT_H

/****************************************************/
#include "lbm_struct.h"
#include "lbm_comm.h"

/****************************************************/
void lbm_init_velocity_0_density_1(lbm_mesh_t * mesh);
void lbm_init_circle_obstacle(lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * comm);
void lbm_init_global_poiseuille_profile(lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type,const lbm_comm_t * comm);
void lbm_init_border(lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * comm);
void lbm_init_mesh_state(lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * comm);
void lbm_init_image_obstacle(lbm_mesh_t * mesh, lbm_mesh_type_t * mesh_type, const lbm_comm_t * mesh_comm,const char * fname);

#endif //LBM_INIT_H
