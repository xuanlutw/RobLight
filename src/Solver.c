#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "BManager.h"
#include "Common.h"
#include "FManager.h"
#include "PGraph.h"
#include "Solver.h"
#include "utils.h"

// ----------------------------------- type ------------------------------------
struct Solver {
    Common   *common;
    PGraph   *G;
    FManager *fm;
    BManager *bm;

    bool robust;
    bool timeout;

    size_t num_nodes;
    size_t num_cuts;
    size_t num_leafs;

    bool   sat;
    bool   flush;
    bool   comp_radius;
    size_t radius;

    bool    set_timeout;
    clock_t clock_start;
    clock_t clock_end;
    clock_t clock_feat;
    clock_t clock_bound;
    clock_t clock_pgraph;
    clock_t clock_profiling;
};

// --------------------------------- lifecycle ---------------------------------
Solver *Solver_alloc(const char *gnn_path, const char *graph_path,
                     const char *feat_path, const char *pert, size_t variant) {
    Solver *self = XMALLOC(sizeof(Solver));

    self->common = Common_alloc(gnn_path, graph_path, feat_path, pert, variant);
    self->G      = PGraph_alloc(self->common);
    self->fm     = FManager_alloc(self->common, self->G);
    self->bm     = BManager_alloc(self->common, self->G);

    return self;
}

void Solver_free(Solver *self) {
    BManager_free(self->bm);
    FManager_free(self->fm);
    PGraph_free(self->G);
    Common_free(self->common);

    free(self);
}

// -------------------------------- statistics ---------------------------------
const char *Solver_result(Solver *self) {
    return self->timeout ? "TIMEOUT" : (self->robust ? "ROBUST" : "NONROBUST");
}

size_t Solver_radius(Solver *self) {
    return self->radius;
}

size_t Solver_ori(Solver *self) {
    return ORI;
}

size_t Solver_tgt(Solver *self) {
    return TGT;
}

size_t Solver_num_nodes(Solver *self) {
    return self->num_nodes;
}

size_t Solver_num_cuts(Solver *self) {
    return self->num_cuts;
}

size_t Solver_num_leafs(Solver *self) {
    return self->num_leafs;
}

double Solver_time(Solver *self) {
    return (double)(self->clock_end - self->clock_start) / CLOCKS_PER_SEC;
}

double Solver_ratio_feat(Solver *self) {
    return (double)(self->clock_feat) / self->clock_profiling;
}

double Solver_ratio_bound(Solver *self) {
    return (double)(self->clock_bound) / self->clock_profiling;
}

double Solver_ratio_pgraph(Solver *self) {
    return (double)(self->clock_pgraph) / self->clock_profiling;
}

void Solver_dump_perturbation(Solver *self, FILE *fp) {
    Common_dump_perturbation(self->common, fp);
}

void Solver_dump_variant(Solver *self, FILE *fp) {
    Common_dump_variant(self->common, fp);
}

// -------------------------------- robustness ---------------------------------
static void Solver_set_statistics(Solver *self) {
    bool profiling = (self->num_nodes % PROF_FREQ == 0);

    PGraph_set_profiling(self->G, profiling);
    FManager_set_profiling(self->fm, profiling);
    BManager_set_profiling(self->bm, profiling);

    self->num_nodes++;
}

static bool Solver_check_timeout(Solver *self) {
    bool profiling = (self->num_nodes % PROF_FREQ == 0);

    if (profiling && self->set_timeout && (clock() > self->clock_end))
        self->timeout = true;

    return self->timeout;
}

static void Solver_push(Solver *self, Edge_type edge_type, size_t v, size_t u,
                        Op_type op_type) {
    PGraph_push(self->G, edge_type, v, u, op_type);
    FManager_push(self->fm);
    BManager_push(self->bm);
}

static void Solver_pop(Solver *self) {
    PGraph_pop(self->G);
    FManager_pop(self->fm);
    BManager_pop(self->bm);
}

static void Solver_sat(Solver *self) {
    if (Solver_check_timeout(self))
        return;

    Solver_set_statistics(self);

    // check non-robust tester
    if (FManager_sat(self->fm)) {
        self->sat = true;
        if (self->comp_radius) {  // set current budget to 0 and continue
            self->flush = true;
            self->radius -= PGraph_budget_down(self->G);
        }
        return;
    }

    // return if it is a leaf node
    if ((PGraph_gbudget(self->G) == 0) || !PGraph_pick_next_edge(self->G)) {
        self->num_leafs++;
        self->num_cuts++;
        return;
    }

    // check bound propagator
    if (BManager_unsat(self->bm)) {
        self->num_cuts++;
        return;
    }

    // keep next edge in stack
    Edge_type edge_type = PGraph_edge_type(self->G);
    size_t    v         = PGraph_v(self->G);
    size_t    u         = PGraph_u(self->G);
    Op_type   op_type;
    bool      flush = false;

    // split
    op_type = (edge_type == PEDGE) ? CUT : CON;
    Solver_push(self, edge_type, v, u, op_type);
    Solver_sat(self);
    Solver_pop(self);
    // explore the other branch if we need to compute the radius
    if (!self->comp_radius && self->sat)
        return;
    if (self->comp_radius && self->flush) {
        BManager_flush(self->bm);  // flush for the other branch
        flush = true;
    }

    op_type = (edge_type == PEDGE) ? CON : CUT;
    Solver_push(self, edge_type, v, u, op_type);
    Solver_sat(self);
    Solver_pop(self);
    if (!self->comp_radius && self->sat)
        return;
    if (self->comp_radius && self->flush)
        flush = true;

    // pass flush to parent
    self->flush = flush;
}

void Solver_check_robust(Solver *self, bool comp_radius, size_t gbudget,
                         size_t lbudget, size_t ori, size_t shift,
                         double timeout) {
    // initialize
    self->num_nodes   = 0;
    self->num_cuts    = 0;
    self->num_leafs   = 0;
    self->timeout     = false;
    self->sat         = false;
    self->flush       = false;
    self->comp_radius = comp_radius;

    // prepare computation of radius
    if (comp_radius) {
        self->radius = N_EDGES;
        gbudget      = N_EDGES;
        lbudget      = N_EDGES;
    }

    // clock start
    self->set_timeout = (timeout > 0);
    self->clock_start = clock();
    self->clock_end   = self->clock_start + timeout * CLOCKS_PER_SEC;

    // initialize components
    Common_init(self->common, ori, shift);
    PGraph_init(self->G, gbudget, lbudget);
    FManager_init(self->fm);
    BManager_init(self->bm);

    // check robustness
    Solver_sat(self);

    // clock end
    self->clock_end = clock();

    // store results
    self->robust       = !self->sat;
    self->radius       = (self->timeout || self->robust) ? 0 : self->radius - 1;
    self->clock_feat   = FManager_clock(self->fm);
    self->clock_bound  = BManager_clock(self->bm);
    self->clock_pgraph = PGraph_clock(self->G);
    self->clock_profiling =
        self->clock_feat + self->clock_bound + self->clock_pgraph;
}
