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
} class_t;

typedef enum {
    AGGR_SUM,
    AGGR_MAX,
    AGGR_MEAN
} aggr_t;

typedef enum {
    GRAPH_DIRECTED,
    GRAPH_UNDIRECTED
} graph_t;

typedef enum {
    PERT_DEL_ONLY,
    PERT_DEL_INS
} pert_t;

typedef enum {
    PEDGE,
    NEDGE,
    QEDGE,
    XEDGE
} edge_t;

typedef enum {
    CUT,
    CON
} op_t;

static inline const char *stringize_op(op_t op) {
    return (op == CUT) ? "CUT" : "CON";
}

// -----------------------------------------------------------------------------
typedef struct HGraph HGraph;

typedef struct {
    // GNN
    class_t class;
    aggr_t aggr;

    size_t  n_layers;
    size_t *dim;
    size_t  dim_max;

    MAT *cC;
    MAT *cA;
    VEC *cb;

    MAT cAP;
    VEC cbP;

    size_t ori;
    size_t tgt;

    // graph
    size_t  n_vertices;
    size_t  n_edges;
    graph_t directed;

    HGraph *input_graph;

    // feature
    MAT input_feat;
    MAT input_feat_rxcA;
    MAT input_feat_rxcC_cb;

    // perturbation
    pert_t pert;

    // options
    bool inc_comp;
    bool reorder_comp;
    bool tight_bound;
    bool heuristic_pick;
} Common;

extern Common common;

// --------------------------------- iterator ----------------------------------
// GNN
#define ITER_LAYERS(l) \
    for (size_t l = 1; l <= common.n_layers; ++l)

#define ITER_LAYERS_EXT(l) \
    for (size_t l = 0; l <= common.n_layers; ++l)

#define IS_LAST_LAYER(l) \
    ((l) == common.n_layers)

#define NOT_LAST_LAYER(l) \
    ((l) < common.n_layers)

#define IS_NODE_CLASS \
    (common.class == CLASS_NODE)

#define IS_GRAPH_CLASS \
    (common.class == CLASS_GRAPH)

#define IS_SUM_GNN \
    (common.aggr == AGGR_SUM)

#define IS_MAX_GNN \
    (common.aggr == AGGR_MAX)

#define IS_MEAN_GNN \
    (common.aggr == AGGR_MEAN)

// graph
#define ITER_VTXS(v) \
    for (size_t v = 0; v < common.n_vertices; ++v)

#define IS_DIRECTED \
    (common.directed == GRAPH_DIRECTED)

#define IS_UNDIRECTED \
    (common.directed == GRAPH_UNDIRECTED)

// perturbation
#define IS_DEL_ONLY \
    (common.pert == PERT_DEL_ONLY)

#define IS_DEL_INS \
    (common.pert == PERT_DEL_INS)

// options
#define USE_INC_COMP \
    (common.inc_comp == true)

#define USE_REORDER_COMP \
    (common.reorder_comp == true)

#define USE_TIGHT_BOUND \
    (common.tight_bound == true)

#define USE_HEURISTIC_PICK \
    (common.heuristic_pick == true)

// ---------------------------------- getter -----------------------------------
// GNN
#define N_LAYERS     (common.n_layers)
#define N_LAYERS_EXT (common.n_layers + 1)
#define DIM          (common.dim)
#define DIM_LAST     (common.dim[N_LAYERS])
#define DIM_MAX      (common.dim_max)
#define CC           (common.cC)
#define CA           (common.cA)
#define CB           (common.cb)
#define CAP          (common.cAP)
#define CBP          (common.cbP)
#define ORI          (common.ori)
#define TGT          (common.tgt)

// graph
#define N_VERTICES  (common.n_vertices)
#define N_EDGES     (common.n_edges)
#define INPUT_GRAPH (common.input_graph)

// feature
#define INPUT_FEAT         (common.input_feat)
#define INPUT_FEAT_RXCA    (common.input_feat_rxcA)
#define INPUT_FEAT_RXCC_CB (common.input_feat_rxcC_cb)

// --------------------------------- lifecycle ---------------------------------
void Common_init(const char *gnn_path, const char *graph_path,
                 const char *feat_path, const char *pert, size_t variant,
                 size_t ori, size_t shift);
void Common_cleanup();
void Common_dump();

// ----------------------------------- info ------------------------------------
void Common_write_perturbation(FILE *fp);
void Common_write_variant(FILE *fp);
