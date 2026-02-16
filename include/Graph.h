#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "BArray.h"
#include "Common.h"

typedef struct Graph {
    Common *common;

    BArray **iadj;
    BArray **oadj;

    BArray **kadj;
} Graph;

// --------------------------------- lifecycle ---------------------------------
Graph *Graph_alloc(Common *common);
Graph *Graph_alloc_fp(Common *common, FILE *fp);
void Graph_free(Graph *self);
void Graph_dump(Graph *self, bool with_knbr);
void Graph_copy(Graph *self, Graph *G);

// ----------------------------- basic operations ------------------------------
static inline bool Graph_is_edge(Graph *self, size_t v, size_t u) {
    return BArray_test(self->iadj[v], u);
}

static inline void Graph_set_edge(Graph *self, size_t v, size_t u) {
    BArray_set(self->iadj[v], u);
    BArray_set(self->oadj[u], v);
}

static inline void Graph_del_edge(Graph *self, size_t v, size_t u) {
    BArray_unset(self->iadj[v], u);
    BArray_unset(self->oadj[u], v);
}

static inline size_t Graph_ideg(Graph *self, size_t v) {
    return BArray_counter(self->iadj[v]);
}

static inline size_t Graph_odeg(Graph *self, size_t v) {
    return BArray_counter(self->oadj[v]);
}

static inline uint16_t *Graph_ilist(Graph *self, size_t v) {
    return BArray_list(self->iadj[v]);
}

static inline uint16_t *Graph_olist(Graph *self, size_t v) {
    return BArray_list(self->oadj[v]);
}

static inline BArray *Graph_iadj(Graph *self, size_t v) {
    return self->iadj[v];
}

static inline BArray *Graph_oadj(Graph *self, size_t v) {
    return self->oadj[v];
}

size_t Graph_count_edges(Graph *self);

// ------------------------- k-hop incoming neighbors --------------------------
static inline bool Graph_is_knbr(Graph *self, size_t k, size_t u) {
    assert(k <= N_LAYERS);

    if (u == 0)
        return true;
    else if (k == 0)
        return false;
    else
        return BArray_test(self->kadj[k], u);
}

static inline size_t Graph_n_knbr(Graph *self, size_t k) {
    return BArray_counter(self->kadj[k]);
}

static inline BArray *Graph_kadj(Graph *self, size_t k) {
    return self->kadj[k];
}

void Graph_update_knbr(Graph *self);
