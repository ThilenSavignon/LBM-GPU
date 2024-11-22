/*****************************************************
    AUTHOR  : Sébastien Valat
    MAIL    : sebastien.valat@univ-grenoble-alpes.fr
    LICENSE : BSD
    YEAR    : 2021
    COURSE  : Parallel Algorithms and Programming
*****************************************************/

/****************************************************/
#ifndef LBM_EXERCICES_H
#define LBM_EXERCICES_H

/****************************************************/
//includes
#include "lbm_comm.h"
#include "lbm_save.h"
#include "lbm_phys.h"

/****************************************************/
//sequential setup
void lbm_comm_init_ex0( lbm_comm_t * comm, int total_width, int total_height );
void lbm_comm_release_ex0( lbm_comm_t * comm );
void lbm_comm_ghost_exchange_ex0(lbm_comm_t * comm, lbm_mesh_t * mesh );
void lbm_save_ex0(lbm_file_mesh_t * save_buffer, lbm_comm_t * comm, lbm_mesh_t * mesh_to_save, lbm_mesh_type_t * mesh_type, int write_step);
void lbm_do_step_ex0(lbm_comm_t * comm, lbm_mesh_type_t * mesh_type, lbm_mesh_t * mesh, lbm_mesh_t * temp_mesh );

/****************************************************/
//linear split
void lbm_comm_init_ex1( lbm_comm_t * comm, int total_width, int total_height );
void lbm_comm_release_ex1( lbm_comm_t * comm );
void lbm_comm_ghost_exchange_ex1(lbm_comm_t * comm, lbm_mesh_t * mesh );
void lbm_save_ex1(lbm_file_mesh_t * save_buffer, lbm_comm_t * comm, lbm_mesh_t * mesh_to_save, lbm_mesh_type_t * mesh_type, int write_step);
void lbm_do_step_ex1(lbm_comm_t * comm, lbm_mesh_type_t * mesh_type, lbm_mesh_t * mesh, lbm_mesh_t * temp_mesh );

/****************************************************/
//odd even on liear comms
void lbm_comm_init_ex2( lbm_comm_t * comm, int total_width, int total_height );
void lbm_comm_release_ex2( lbm_comm_t * comm );
void lbm_comm_ghost_exchange_ex2(lbm_comm_t * comm, lbm_mesh_t * mesh );
void lbm_save_ex2(lbm_file_mesh_t * save_buffer, lbm_comm_t * comm, lbm_mesh_t * mesh_to_save, lbm_mesh_type_t * mesh_type, int write_step);
void lbm_do_step_ex2(lbm_comm_t * comm, lbm_mesh_type_t * mesh_type, lbm_mesh_t * mesh, lbm_mesh_t * temp_mesh );

/****************************************************/
//async
void lbm_comm_init_ex3( lbm_comm_t * comm, int total_width, int total_height );
void lbm_comm_release_ex3( lbm_comm_t * comm );
void lbm_comm_ghost_exchange_ex3(lbm_comm_t * comm, lbm_mesh_t * mesh );
void lbm_save_ex3(lbm_file_mesh_t * save_buffer, lbm_comm_t * comm, lbm_mesh_t * mesh_to_save, lbm_mesh_type_t * mesh_type, int write_step);
void lbm_do_step_ex3(lbm_comm_t * comm, lbm_mesh_type_t * mesh_type, lbm_mesh_t * mesh, lbm_mesh_t * temp_mesh );

/****************************************************/
//2D
void lbm_comm_init_ex4( lbm_comm_t * comm, int total_width, int total_height );
void lbm_comm_release_ex4( lbm_comm_t * comm );
void lbm_comm_ghost_exchange_ex4(lbm_comm_t * comm, lbm_mesh_t * mesh );
void lbm_save_ex4(lbm_file_mesh_t * save_buffer, lbm_comm_t * comm, lbm_mesh_t * mesh_to_save, lbm_mesh_type_t * mesh_type, int write_step);
void lbm_do_step_ex4(lbm_comm_t * comm, lbm_mesh_type_t * mesh_type, lbm_mesh_t * mesh, lbm_mesh_t * temp_mesh );

/****************************************************/
//2D + mpi types
void lbm_comm_init_ex5( lbm_comm_t * comm, int total_width, int total_height );
void lbm_comm_release_ex5( lbm_comm_t * comm );
void lbm_comm_ghost_exchange_ex5(lbm_comm_t * comm, lbm_mesh_t * mesh );
void lbm_save_ex5(lbm_file_mesh_t * save_buffer, lbm_comm_t * comm, lbm_mesh_t * mesh_to_save, lbm_mesh_type_t * mesh_type, int write_step);
void lbm_do_step_ex5(lbm_comm_t * comm, lbm_mesh_type_t * mesh_type, lbm_mesh_t * mesh, lbm_mesh_t * temp_mesh );

/****************************************************/
//async
void lbm_comm_init_ex6( lbm_comm_t * comm, int total_width, int total_height );
void lbm_comm_release_ex6( lbm_comm_t * comm );
void lbm_comm_ghost_exchange_ex6(lbm_comm_t * comm, lbm_mesh_t * mesh );
void lbm_save_ex6(lbm_file_mesh_t * save_buffer, lbm_comm_t * comm, lbm_mesh_t * mesh_to_save, lbm_mesh_type_t * mesh_type, int write_step);
void lbm_do_step_ex6(lbm_comm_t * comm, lbm_mesh_type_t * mesh_type, lbm_mesh_t * mesh, lbm_mesh_t * temp_mesh );

/****************************************************/
//select
void lbm_comm_init_ex_select( lbm_comm_t * comm, int total_width, int total_height );
void lbm_comm_release_ex_select( lbm_comm_t * comm );
void lbm_comm_ghost_exchange_ex_select(lbm_comm_t * comm, lbm_mesh_t * mesh );
void lbm_save_ex_select(lbm_file_mesh_t * save_buffer, lbm_comm_t * comm, lbm_mesh_t * mesh_to_save, lbm_mesh_type_t * mesh_type, int write_step);
void lbm_do_step_ex_select(lbm_comm_t * comm, lbm_mesh_type_t * mesh_type, lbm_mesh_t * mesh, lbm_mesh_t * temp_mesh );

/****************************************************/
//selector
void lbm_ex_select(int id);

#endif //LBM_EXERCICES_H
