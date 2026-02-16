#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "Matrix.h"

// ----------------------------------- type ------------------------------------
typedef enum {
    CLASS_NODE,
    CLASS_GRAPH
} Class_type;

typedef enum {
    AGGR_SUM,
    AGGR_MAX,
    AGGR_MEAN
} Aggr_type;

typedef enum {
    GRAPH_DIRECTED,
    GRAPH_UNDIRECTED
} Graph_type;

typedef enum {
    PERT_DEL_ONLY,
    PERT_DEL_INS
} Pert_type;

typedef enum {
    PEDGE,
    QEDGE
} Edge_type;

typedef enum {
    CUT,
    CON
} Op_type;

typedef struct Graph  Graph;
typedef struct Common Common;

struct Common {
    Common *common;

    // GNN
    Class_type class;
    Aggr_type aggr;

    size_t  n_layers;
    size_t *dim;
    size_t  dimL;

    MAT **cC;
    MAT **cA;
    VEC **cb;

    MAT *cAL;
    VEC *cbL;

    MAT **cC_abs;
    MAT **cA_abs;

    size_t ori;
    size_t tgt;

    // Graph
    size_t     n_vertices;
    size_t     n_edges;
    Graph_type directed;

    Graph *G;

    MAT *input_feat;
    MAT *input_feat_rxcA;
    MAT *input_feat_rxcC_cb;

    // Perturbation
    Pert_type pert;

    // Options
    bool inc_comp;
    bool reorder_comp;
    bool tight_bound;
    bool heuristic_pick;
};

// --------------------------------- lifecycle ---------------------------------
Common *Common_alloc(const char *gnn_path, const char *graph_path,
                     const char *feat_path, const char *pert, size_t variant);
void Common_free(Common *self);
void Common_dump(Common *self);

// --------------------------------- iterator ----------------------------------
// GNN
#define ITER_LAYERS(l) \
    for (size_t l = 1; l <= self->common->n_layers; ++l)

#define ITER_LAYERS_EXT(l) \
    for (size_t l = 0; l <= self->common->n_layers; ++l)

#define IS_LAST_LAYER(l) \
    ((l) == self->common->n_layers)

#define NOT_LAST_LAYER(l) \
    ((l) < self->common->n_layers)

#define IS_NODE_CLASS \
    (self->common->class == CLASS_NODE)

#define IS_GRAPH_CLASS \
    (self->common->class == CLASS_GRAPH)

#define IS_SUM_GNN \
    (self->common->aggr == AGGR_SUM)

#define IS_MAX_GNN \
    (self->common->aggr == AGGR_MAX)

#define IS_MEAN_GNN \
    (self->common->aggr == AGGR_MEAN)

// Graph
#define ITER_VTXS(v) \
    for (size_t v = 0; v < self->common->n_vertices; ++v)

#define IS_DIRECTED \
    (self->common->directed == GRAPH_DIRECTED)

#define IS_UNDIRECTED \
    (self->common->directed == GRAPH_UNDIRECTED)

// Perturbation
#define IS_DEL_ONLY \
    (self->common->pert == PERT_DEL_ONLY)

#define IS_DEL_INS \
    (self->common->pert == PERT_DEL_INS)

// Options
#define USE_INC_COMP \
    (self->common->inc_comp == true)

#define USE_REORDER_COMP \
    (self->common->reorder_comp == true)

#define USE_TIGHT_BOUND \
    (self->common->tight_bound == true)

#define USE_HEURISTIC_PICK \
    (self->common->heuristic_pick == true)

// ---------------------------------- getter -----------------------------------
// GNN
#define N_LAYERS     (self->common->n_layers)
#define N_LAYERS_EXT (self->common->n_layers + 1)
#define DIM          (self->common->dim)
#define DIM_LAST     (self->common->dim[N_LAYERS])
#define DIML         (self->common->dimL)
#define CC           (self->common->cC)
#define CC_ABS       (self->common->cC_abs)
#define CA           (self->common->cA)
#define CA_ABS       (self->common->cA_abs)
#define CB           (self->common->cb)
#define CAL          (self->common->cAL)
#define CBL          (self->common->cbL)
#define ORI          (self->common->ori)
#define TGT          (self->common->tgt)

// Graph
#define N_VERTICES         (self->common->n_vertices)
#define N_EDGES            (self->common->n_edges)
#define INPUT_G            (self->common->G)
#define INPUT_FEAT         (self->common->input_feat)
#define INPUT_FEAT_RXCA    (self->common->input_feat_rxcA)
#define INPUT_FEAT_RXCC_CB (self->common->input_feat_rxcC_cb)

// ----------------------------------- info ------------------------------------
void Common_dump_perturbation(Common *self, FILE *fp);
void Common_dump_variant(Common *self, FILE *fp);

// ---------------------------------- status -----------------------------------
void Common_init(Common *self, size_t ori, size_t shift);
