#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

#define PROF_FREQ 65536

// ----------------------------------- type ------------------------------------
typedef struct Solver Solver;

// --------------------------------- lifecycle ---------------------------------
Solver *Solver_alloc(const char *gnn_path, const char *graph_path,
                     const char *feat_path, const char *pert, size_t variant);
void Solver_free(Solver *self);

// -------------------------------- statistics ---------------------------------
const char *Solver_result(Solver *self);
size_t Solver_radius(Solver *self);
size_t Solver_ori(Solver *self);
size_t Solver_tgt(Solver *self);
size_t Solver_num_nodes(Solver *self);
size_t Solver_num_cuts(Solver *self);
size_t Solver_num_leafs(Solver *self);
double Solver_time(Solver *self);
double Solver_ratio_feat(Solver *self);
double Solver_ratio_bound(Solver *self);
double Solver_ratio_pgraph(Solver *self);
void Solver_dump_perturbation(Solver *self, FILE *fp);
void Solver_dump_variant(Solver *self, FILE *fp);

// -------------------------------- robustness ---------------------------------
void Solver_check_robust(Solver *self, bool comp_radius, size_t gbudget,
                         size_t lbudget, size_t ori, size_t shift,
                         double timeout);
