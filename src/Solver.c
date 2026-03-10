#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "BManager.h"
#include "Common.h"
#include "FManager.h"
#include "GManager.h"
#include "Solver.h"
#include "utils.h"

// ----------------------------------- type ------------------------------------
typedef enum {
    SAT,
    UNSAT,
    TIMEOUT
} Status;

struct Solver {
    GManager *gm;
    FManager *fm;
    BManager *bm;

    Status status;
    bool   comp_radius;
    size_t radius;
    size_t rem_budget;

    size_t num_nodes;
    size_t num_cuts;
    size_t num_leafs;

    bool    set_timeout;
    clock_t clock_start;
    clock_t clock_end;
    clock_t clock_feat;
    clock_t clock_bound;
    clock_t clock_graph;
    clock_t clock_profiling;
};

// --------------------------------- lifecycle ---------------------------------
Solver *Solver_alloc(const char *gnn_path, const char *graph_path,
                     const char *feat_path, const char *pert, size_t variant,
                     bool comp_radius, size_t gbudget, size_t lbudget,
                     size_t ori, size_t shift) {
    Solver *self = XMALLOC(sizeof(Solver));

    Common_init(gnn_path, graph_path, feat_path, pert, variant, ori, shift);

    self->gm = GManager_alloc(comp_radius, gbudget, lbudget);
    self->fm = FManager_alloc(self->gm);
    self->bm = BManager_alloc(self->gm);

    self->status      = UNSAT;
    self->comp_radius = comp_radius;
    self->radius      = N_EDGES;

    self->num_nodes = 0;
    self->num_cuts  = 0;
    self->num_leafs = 0;

    return self;
}

void Solver_free(Solver *self) {
    GManager_free(self->gm);
    FManager_free(self->fm);
    BManager_free(self->bm);

    Common_cleanup();

    free(self);
}

// -------------------------------- statistics ---------------------------------
const char *Solver_result(Solver *self) {
    switch (self->status) {
    case SAT:
        return "NONROBUST";
    case UNSAT:
        return "ROBUST";
    case TIMEOUT:
        return "TIMEOUT";
    default:
        CHECK(true, "never!");
    }
}

size_t Solver_radius(Solver *self) {
    return self->radius;
}

size_t Solver_ori(Solver *self __attribute__((unused))) {
    return ORI;
}

size_t Solver_tgt(Solver *self __attribute__((unused))) {
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

double Solver_ratio_graph(Solver *self) {
    return (double)(self->clock_graph) / self->clock_profiling;
}

void Solver_write_perturbation(Solver *self __attribute__((unused)), FILE *fp) {
    Common_write_perturbation(fp);
}

void Solver_write_variant(Solver *self __attribute__((unused)), FILE *fp) {
    Common_write_variant(fp);
}

// -------------------------------- robustness ---------------------------------
static bool Solver_set_statistics(Solver *self) {
    // set profiling
    bool profiling = (self->num_nodes % PROF_FREQ == 0);
    GManager_set_profiling(self->gm, profiling);
    FManager_set_profiling(self->fm, profiling);
    BManager_set_profiling(self->bm, profiling);

    // increase counter
    self->num_nodes++;

    // check timeout
    return (profiling && self->set_timeout && (clock() > self->clock_end));
}

static void Solver_push(Solver *self, edge_t edge, size_t v, size_t u,
                        op_t op) {
    GManager_push(self->gm, edge, v, u, op);
    if (USE_INC_COMP) {
        FManager_push(self->fm);
        BManager_push(self->bm);
    }
}

static void Solver_pop(Solver *self) {
    GManager_pop(self->gm);
    if (USE_INC_COMP) {
        FManager_pop(self->fm);
        BManager_pop(self->bm);
    }
}

#ifndef NDEBUG
#define INDENT(depth)                        \
    do {                                     \
        for (size_t i = 0; i < (depth); ++i) \
             printf("   ");                  \
    } while(0)

#define LOG(depth, msg, ...)             \
    do {                                 \
        INDENT(depth);                   \
        printf(msg "\n", ##__VA_ARGS__); \
    } while(0)
#else
#define LOG(depth, msg, ...) (void)0
#endif

static Status Solver_sat(Solver *self, size_t depth) {
#ifndef NDEBUG
    // GManager_dump(self->gm);
#endif
    LOG(depth, "=> iter %ld (%ld)", self->num_nodes,
        GManager_gbudget(self->gm));

    // set statistics and check timeout
    if (Solver_set_statistics(self)) {
        LOG(depth, " > STATUS: timeout");
        return TIMEOUT;
    }

    // update nbrs in the grounding graph for non-robust tester and edge picker
    if (IS_NODE_CLASS && (USE_HEURISTIC_PICK || USE_INC_COMP))
        GManager_update_ginbr(self->gm);

    // check non-robust tester
    if (FManager_sat(self->fm)) {
        LOG(depth, " > STATUS: find counter example");
        self->rem_budget = GManager_gbudget(self->gm);
        return SAT;
    }

    // return if there is no budget
    if (GManager_gbudget(self->gm) == 0) {
        LOG(depth, " > STATUS: leaf node (no budget)");
        self->num_leafs++;
        self->num_cuts++;
        return UNSAT;
    }

    // return if there is no next edge
    edge_t edge;
    size_t v, u;
    if (!GManager_pick_next_edge(self->gm, &edge, &v, &u)) {
        LOG(depth, " > STATUS: leaf node (no next edge)");
        self->num_leafs++;
        self->num_cuts++;
        return UNSAT;
    }

    // update nbrs in the adjusted graph for bound propagator
    if (IS_NODE_CLASS && USE_INC_COMP)
        GManager_update_ainbr(self->gm);

    // check bound propagator
    if (BManager_unsat(self->bm)) {
        LOG(depth, " > STATUS: bound confilct");
        self->num_cuts++;
        return UNSAT;
    }

    // split
    op_t op = (edge == PEDGE) ? CUT : CON;
    LOG(depth, " > STATUS: %s pedge (%ld, %ld)", stringize_op(op), v, u);

    Solver_push(self, edge, v, u, op);
    Status ret = Solver_sat(self, depth + 1);
    Solver_pop(self);
    if (ret == TIMEOUT)
        return TIMEOUT;
    else if (ret == SAT)
        return SAT;

    op = (edge == PEDGE) ? CON : CUT;
    LOG(depth, " > STATUS: %s pedge (%ld, %ld)", stringize_op(op), v, u);

    Solver_push(self, edge, v, u, op);
    ret = Solver_sat(self, depth + 1);
    Solver_pop(self);
    if (ret == TIMEOUT)
        return TIMEOUT;
    else if (ret == SAT)
        return SAT;

    return UNSAT;
}

void Solver_check_robust(Solver *self, double timeout) {
#ifndef NDEBUG
    // Common_dump();
#endif

    // clock start
    self->set_timeout = (timeout > 0);
    self->clock_start = clock();
    self->clock_end   = self->clock_start + timeout * CLOCKS_PER_SEC;

    // check robustness
    if (self->comp_radius) {
        Status status_all = UNSAT;
        while (true) {
            Status status = Solver_sat(self, 0);

            if (status == TIMEOUT) {
                status_all = TIMEOUT;
                break;
            }
            else if (status == UNSAT) {
                break;
            }
            else {
                status_all = SAT;

                if (self->rem_budget == self->radius) {
                    self->radius = 0;
                    break;
                }
                else {
                    self->radius -= (self->rem_budget + 1);
                    GManager_budget_down(self->gm, self->rem_budget + 1);
                }
            }
        }
        self->radius = (status_all == SAT) ? self->radius : 0;
        self->status = status_all;
    }
    else {
        self->status = Solver_sat(self, 0);
    }

    // clock end
    self->clock_end = clock();

    // store results
    self->clock_feat  = FManager_clock(self->fm);
    self->clock_bound = BManager_clock(self->bm);
    self->clock_graph = GManager_clock(self->gm);
    self->clock_profiling =
        self->clock_feat + self->clock_bound + self->clock_graph;
}
