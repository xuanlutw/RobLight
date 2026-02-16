#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#include "BArray.h"
#include "BudManager.h"
#include "Common.h"
#include "Graph.h"
#include "Stack.h"

typedef struct {
    Common *common;

    Graph *GP;
    Graph *GQ;
    Graph *GN;
    Graph *G;

    Edge_type edge_type;
    size_t    v;
    size_t    u;
    Op_type   op_type;

    BArray *updated;

    BudManager *budm;

    Stack *stack;

    bool    profiling;
    clock_t clock;
} PGraph;

// --------------------------------- lifecycle ---------------------------------
PGraph *PGraph_alloc(Common *common);
void PGraph_free(PGraph *self);
void PGraph_dump(PGraph *self);

// ----------------------------- basic operations ------------------------------
static inline size_t PGraph_pideg(PGraph *self, size_t v) {
    return Graph_ideg(self->GP, v);
}

static inline size_t PGraph_qideg(PGraph *self, size_t v) {
    return Graph_ideg(self->GQ, v);
}

static inline size_t PGraph_nideg(PGraph *self, size_t v) {
    return Graph_ideg(self->GN, v);
}

static inline size_t PGraph_ideg(PGraph *self, size_t v) {
    return Graph_ideg(self->G, v);
}

static inline uint16_t *PGraph_ilist(PGraph *self, size_t v) {
    return Graph_ilist(self->G, v);
}

static inline BArray *PGraph_piadj(PGraph *self, size_t v) {
    return Graph_iadj(self->GP, v);
}

static inline BArray *PGraph_qiadj(PGraph *self, size_t v) {
    return Graph_iadj(self->GQ, v);
}

static inline BArray *PGraph_niadj(PGraph *self, size_t v) {
    return Graph_iadj(self->GN, v);
}

static inline BArray *PGraph_iadj(PGraph *self, size_t v) {
    return Graph_iadj(self->G, v);
}

static inline BArray *PGraph_oadj(PGraph *self, size_t v) {
    return Graph_oadj(self->G, v);
}

#define ITER_PINBRS(self, v, u) \
    ITER_BArray(Graph_iadj(self->GP, v), u)

#define ITER_QINBRS(self, v, u) \
    ITER_BArray(Graph_iadj(self->GQ, v), u)

// ------------------------------ k-hop neighbors ------------------------------
static inline bool PGraph_is_knbr(PGraph *self, size_t k, size_t u) {
    return Graph_is_knbr(self->G, k, u);
}

#define ITER_KNBRS(self, k, u) \
    ITER_BArray(Graph_kadj(self->G, k), u)

// --------------------------------- next edge ---------------------------------
bool PGraph_pick_next_edge(PGraph *self);

static inline Edge_type PGraph_edge_type(PGraph *self) {
    return self->edge_type;
}

static inline size_t PGraph_v(PGraph *self) {
    return self->v;
}

static inline size_t PGraph_u(PGraph *self) {
    return self->u;
}

static inline Op_type PGraph_op_type(PGraph *self) {
    return self->op_type;
}

// -------------------------------- statistics ---------------------------------
static inline void PGraph_set_profiling(PGraph *self, bool profiling) {
    self->profiling = profiling;
}

static inline clock_t PGraph_clock(PGraph *self) {
    return self->clock;
}

// ---------------------------------- updated ----------------------------------
static inline BArray *PGraph_updated_edges(PGraph *self) {
    return self->updated;
}

static inline BArray *PGraph_updated_budgets(PGraph *self) {
    return BudManager_updated(self->budm);
}

// ---------------------------------- budgets ----------------------------------
static inline size_t PGraph_gbudget(PGraph *self) {
    return BudManager_gbudget(self->budm);
}

static inline size_t PGraph_lbudget(PGraph *self, size_t v) {
    return BudManager_lbudget(self->budm, v);
}

static inline size_t PGraph_budget_down(PGraph *self) {
    return BudManager_down(self->budm);
}

// ---------------------------------- status -----------------------------------
void PGraph_init(PGraph *self, size_t gbudget, size_t lbudget);
void PGraph_push(PGraph *self, Edge_type edge_type, size_t v, size_t u,
                 Op_type op_type);
void PGraph_pop(PGraph *self);
