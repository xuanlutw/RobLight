#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>

#include "BudManager.h"
#include "Common.h"
#include "Graph.h"
#include "Stack.h"

typedef struct {
    HGraph inc_graph;
    HGraph out_graph;

    edge_t edge;
    size_t v;
    size_t u;
    op_t   op;

    bool     *flag_b;
    uint16_t *queue_b;
    uint16_t *end_b;
    uint16_t *lens_b;

    bool     *flag_f;
    uint16_t *queue_f;
    uint16_t *lens_f;

    BudManager budm;

    Stack stack;

    bool    profiling;
    clock_t clock;
} GManager;

// --------------------------------- lifecycle ---------------------------------
GManager *GManager_alloc(bool comp_radius, size_t gbudget, size_t lbudget);
void GManager_free(GManager *self);
void GManager_dump(GManager *self);

// ----------------------------- basic operations ------------------------------
static inline size_t GManager_pideg(GManager *self, size_t v) {
    return HGraph_pdeg(&self->inc_graph, v);
}

static inline size_t GManager_nideg(GManager *self, size_t v) {
    return HGraph_ndeg(&self->inc_graph, v);
}

static inline size_t GManager_qideg(GManager *self, size_t v) {
    return HGraph_qdeg(&self->inc_graph, v);
}

static inline size_t GManager_gideg(GManager *self, size_t v) {
    return HGraph_gdeg(&self->inc_graph, v);
}

static inline uint16_t *GManager_pilist(GManager *self, size_t v) {
    return HGraph_plist(&self->inc_graph, v);
}

static inline uint16_t *GManager_nilist(GManager *self, size_t v) {
    return HGraph_nlist(&self->inc_graph, v);
}

static inline uint16_t *GManager_qilist(GManager *self, size_t v) {
    return HGraph_qlist(&self->inc_graph, v);
}

static inline uint16_t *GManager_gilist(GManager *self, size_t v) {
    return HGraph_glist(&self->inc_graph, v);
}

#define ITER_PINBRS(self, v, u) \
    ITER_PNBRS(&self->inc_graph, v, u)

#define ITER_QINBRS(self, v, u) \
    ITER_QNBRS(&self->inc_graph, v, u)

// --------------------------------- next edge ---------------------------------
bool GManager_pick_next_edge(GManager *self, edge_t *edge, size_t *vn,
                             size_t *un);

// -------------------------------- statistics ---------------------------------
static inline void GManager_set_profiling(GManager *self, bool profiling) {
    self->profiling = profiling;
}

static inline clock_t GManager_clock(GManager *self) {
    return self->clock;
}

// ---------------------------------- update -----------------------------------
void GManager_comp_index_f(GManager *self, uint16_t **index,
                           uint16_t *counter_self, uint16_t *counter_nbrs);

void GManager_comp_index_b(GManager *self, uint16_t **index,
                           uint16_t *counter_self, uint16_t *counter_nbrs);

// ---------------------------------- budgets ----------------------------------
static inline size_t GManager_gbudget(GManager *self) {
    return BudManager_gbudget(&self->budm);
}

static inline size_t GManager_lbudget(GManager *self, size_t v) {
    return BudManager_lbudget(&self->budm, v);
}

static inline void GManager_budget_down(GManager *self, uint16_t delta) {
    BudManager_down(&self->budm, delta);
}

// ---------------------------------- status -----------------------------------
static inline void GManager_update_ginbr(GManager *self) {
    HGraph_update_gnbr(&self->inc_graph);
}

static inline void GManager_update_ainbr(GManager *self) {
    HGraph_update_anbr(&self->inc_graph);
}

void GManager_push(GManager *self, edge_t edge, size_t v, size_t u, op_t op);
void GManager_pop(GManager *self);
