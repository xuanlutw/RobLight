#include <assert.h>
#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "BArray.h"
#include "Common.h"
#include "FManager.h"
#include "Matrix.h"
#include "PGraph.h"
#include "Stack.h"
#include "utils.h"

// ----------------------------------- snap ------------------------------------
typedef struct {
    Node node;

    size_t l;
    size_t v;

    bool push_self;
    bool push_nbrs;

    VEC *feat_v;
    VEC *feat_self_v;
    VEC *feat_nbrs_v;
    VEC *feat_rxcA_v;
} FManagerSnap;

typedef struct {
    Node node;

    VEC *pool;
} FManagerPoolSnap;

static void FManager_snap_free(Node *node) {
    FManagerSnap *snap = (FManagerSnap *)node;

    if (((Node *)snap)->type > 0) {
        VEC_free(snap->feat_v);
        VEC_free(snap->feat_self_v);
        VEC_free(snap->feat_nbrs_v);
        if (snap->feat_rxcA_v != NULL)
            VEC_free(snap->feat_rxcA_v);
    }

    free(snap);
}

static void FManager_poolsnap_free(Node *node) {
    FManagerPoolSnap *snap = (FManagerPoolSnap *)node;

    if (snap->pool != NULL)
        VEC_free(snap->pool);

    free(snap);
}

// ----------------------------------- type ------------------------------------
struct FManager {
    Common *common;
    PGraph *G;

    MAT **feat;
    MAT **feat_self;
    MAT **feat_nbrs;
    MAT **feat_rxcA;
    VEC **tmp_self;
    VEC **tmp_nbrs;
    VEC  *pool;

    BArray *dirty_self;
    BArray *dirty_self_next;
    BArray *dirty_nbrs;
    BArray *dirty_nbrs_next;

    Stack *stack;
    Stack *stack_pool;

    bool init;

    bool    profiling;
    clock_t clock;
};

// --------------------------------- lifecycle ---------------------------------
FManager *FManager_alloc(Common *common, PGraph *G) {
    FManager *self = XMALLOC(sizeof(FManager));

    self->common = common;
    self->G      = G;

    return self;
}

void FManager_free(FManager *self) {
    ITER_LAYERS(l) {
        MAT_free(self->feat[l]);
        MAT_free(self->feat_self[l]);
        MAT_free(self->feat_nbrs[l]);
        if (l == 1)
            continue;
        MAT_free(self->feat_rxcA[l]);
    }
    ITER_LAYERS_EXT(l) {
        VEC_free(self->tmp_self[l]);
        VEC_free(self->tmp_nbrs[l]);
    }
    VEC_free(self->pool);

    free(self->feat);
    free(self->feat_self);
    free(self->feat_nbrs);
    free(self->feat_rxcA);
    free(self->tmp_self);
    free(self->tmp_nbrs);

    BArray_free(self->dirty_self);
    BArray_free(self->dirty_self_next);
    BArray_free(self->dirty_nbrs);
    BArray_free(self->dirty_nbrs_next);

    Stack_free(self->stack);
    Stack_free(self->stack_pool);

    free(self);
}

// ---------------------------------- getter -----------------------------------
MAT *FManager_feat(FManager *self, size_t l) {
    return self->feat[l];
}

// -------------------------------- statistics ---------------------------------
void FManager_set_profiling(FManager *self, bool profiling) {
    self->profiling = profiling;
}

clock_t FManager_clock(FManager *self) {
    return self->clock;
}

// ----------------------------------- snap ------------------------------------
static void FManager_push_snap(FManager *self, size_t l, size_t v,
                               bool push_self, bool push_nbrs) {
    FManagerSnap *snap = (FManagerSnap *)Stack_push(self->stack, l);

    if (Stack_fresh(self->stack)) {
        snap->feat_v      = VEC_alloc(DIM[l]);
        snap->feat_self_v = VEC_alloc(DIM[l]);
        snap->feat_nbrs_v = VEC_alloc(DIM[l]);
        if (NOT_LAST_LAYER(l))
            snap->feat_rxcA_v = VEC_alloc(DIM[l + 1]);
        else
            snap->feat_rxcA_v = NULL;
    }

    snap->l         = l;
    snap->v         = v;
    snap->push_self = push_self;
    snap->push_nbrs = push_nbrs;
    MV_SWAP(snap->feat_v, self->feat[l], v);
    if (push_self)
        MV_SWAP(snap->feat_self_v, self->feat_self[l], v);
    if (push_nbrs)
        MV_SWAP(snap->feat_nbrs_v, self->feat_nbrs[l], v);
    if (NOT_LAST_LAYER(l))
        MV_SWAP(snap->feat_rxcA_v, self->feat_rxcA[l + 1], v);
}

// ------------------------------ comp feat nbrs -------------------------------
static void FManager_comp_nbrs_sum(FManager *self, size_t l, size_t v) {
    if (PGraph_ideg(self->G, v) == 0) {
        VEC_clear(self->tmp_nbrs[l]);
    }
    else {
        MV_acc(self->tmp_nbrs[l], NULL, self->feat_rxcA[l],
               PGraph_iadj(self->G, v));
    }
}

static void FManager_comp_nbrs_max(FManager *self, size_t l, size_t v) {
    if (PGraph_ideg(self->G, v) == 0) {
        VEC_clear(self->tmp_nbrs[l]);
    }
    else {
        MV_max(self->tmp_nbrs[l - 1], NULL, self->feat[l - 1],
               PGraph_iadj(self->G, v));
        MV_dot(self->tmp_nbrs[l], CA[l], self->tmp_nbrs[l - 1]);
    }
}

static void FManager_comp_nbrs_mean(FManager *self, size_t l, size_t v) {
    if (PGraph_ideg(self->G, v) == 0) {
        VEC_clear(self->tmp_nbrs[l]);
    }
    else {
        MV_acc(self->tmp_nbrs[l], NULL, self->feat_rxcA[l],
               PGraph_iadj(self->G, v));
        VEC_div(self->tmp_nbrs[l], PGraph_ideg(self->G, v));
    }
}

// --------------------------------- comp feat ---------------------------------
static void FManager_comp_self(FManager *self, size_t l, size_t v) {
    if (l == 1) {
        VEC_copy(self->tmp_self[l], MSLICE(INPUT_FEAT_RXCC_CB, v));
    }
    else {
        VEC *feat_vp = MSLICE(self->feat[l - 1], v);
        MV_dot(self->tmp_self[l], CC[l], feat_vp);
        VV_add(self->tmp_self[l], self->tmp_self[l], CB[l]);
    }
}

static void FManager_comp_feat(FManager *self, size_t l, size_t v) {
    VEC *feat_v      = MSLICE(self->feat[l], v);
    VEC *feat_self_v = MSLICE(self->feat_self[l], v);
    VEC *feat_nbrs_v = MSLICE(self->feat_nbrs[l], v);

    VV_add(feat_v, feat_self_v, feat_nbrs_v);

    if (NOT_LAST_LAYER(l)) {
        VEC *feat_rxcA_vn = MSLICE(self->feat_rxcA[l + 1], v);
        VEC_relu(feat_v);
        MV_dot(feat_rxcA_vn, CA[l + 1], feat_v);
    }
}

static void FManager_comp(FManager *self) {
    ITER_LAYERS(l) {
        ITER_VTXS(v) {
            FManager_comp_self(self, l, v);
            if (IS_SUM_GNN)
                FManager_comp_nbrs_sum(self, l, v);
            if (IS_MAX_GNN)
                FManager_comp_nbrs_max(self, l, v);
            if (IS_MEAN_GNN)
                FManager_comp_nbrs_mean(self, l, v);
            MV_SWAP(self->tmp_self[l], self->feat_self[l], v);
            MV_SWAP(self->tmp_nbrs[l], self->feat_nbrs[l], v);
            FManager_comp_feat(self, l, v);
        }
    }

    if (IS_GRAPH_CLASS) {
        MV_acc(self->pool, NULL, self->feat[N_LAYERS], NULL);
        VV_add(self->pool, self->pool, CBL);
    }
}

// -------------------------------- update feat --------------------------------
static void FManager_update_single(FManager *self, size_t l, size_t v) {
    bool is_dirty_self = BArray_test(self->dirty_self, v);
    bool is_dirty_nbrs = BArray_test(self->dirty_nbrs, v);
    if (!is_dirty_self && !is_dirty_nbrs)
        return;

    // comp
    if (is_dirty_self)
        FManager_comp_self(self, l, v);
    if (is_dirty_nbrs && IS_SUM_GNN)
        FManager_comp_nbrs_sum(self, l, v);
    if (is_dirty_nbrs && IS_MAX_GNN)
        FManager_comp_nbrs_max(self, l, v);
    if (is_dirty_nbrs && IS_MEAN_GNN)
        FManager_comp_nbrs_mean(self, l, v);

    // push
    if (IS_GRAPH_CLASS && IS_LAST_LAYER(l))
        VV_sub(self->pool, self->pool, MSLICE(self->feat[l], v));
    FManager_push_snap(self, l, v, is_dirty_self, is_dirty_nbrs);

    // swap and comp
    if (is_dirty_self)
        MV_SWAP(self->tmp_self[l], self->feat_self[l], v);
    if (is_dirty_nbrs)
        MV_SWAP(self->tmp_nbrs[l], self->feat_nbrs[l], v);
    FManager_comp_feat(self, l, v);

    // update pool
    if (IS_GRAPH_CLASS && IS_LAST_LAYER(l))
        VV_add(self->pool, self->pool, MSLICE(self->feat[l], v));

    // propagate
    if (NOT_LAST_LAYER(l)) {
        BArray_set(self->dirty_self_next, v);
        BArray_union(self->dirty_nbrs_next, PGraph_oadj(self->G, v));
    }
}

static void FManager_update(FManager *self) {
    ITER_LAYERS(l) {
        // init dirty vertices
        if (l == 1) {
            BArray_clear(self->dirty_self);
            BArray_clear(self->dirty_nbrs);
        }
        else {
            BArray *tmp;
            tmp                   = self->dirty_self;
            self->dirty_self      = self->dirty_self_next;
            self->dirty_self_next = tmp;
            tmp                   = self->dirty_nbrs;
            self->dirty_nbrs      = self->dirty_nbrs_next;
            self->dirty_nbrs_next = tmp;
        }
        BArray_clear(self->dirty_self_next);
        BArray_clear(self->dirty_nbrs_next);

        // set dirty vertices
        Edge_type edge_type = PGraph_edge_type(self->G);
        size_t    v         = PGraph_v(self->G);
        size_t    u         = PGraph_u(self->G);
        Op_type   op_type   = PGraph_op_type(self->G);
        if (IS_DIRECTED) {
            if ((edge_type == PEDGE) && (op_type == CUT))
                BArray_set(self->dirty_nbrs, v);
            if ((edge_type == QEDGE) && (op_type == CON))
                BArray_set(self->dirty_nbrs, v);
        }
        if (IS_UNDIRECTED) {
            if ((edge_type == PEDGE) && (op_type == CUT)) {
                BArray_set(self->dirty_nbrs, v);
                BArray_set(self->dirty_nbrs, u);
            }
            if ((edge_type == QEDGE) && (op_type == CON)) {
                BArray_set(self->dirty_nbrs, v);
                BArray_set(self->dirty_nbrs, u);
            }
        }

        // update features
        if (IS_NODE_CLASS) {
            ITER_KNBRS(self->G, N_LAYERS - l, v) {
                FManager_update_single(self, l, v);
            }
        }
        if (IS_GRAPH_CLASS) {
            ITER_VTXS(v) {
                FManager_update_single(self, l, v);
            }
        }
    }
}

// ---------------------------------- status -----------------------------------
void FManager_init(FManager *self) {
    // allocate
    self->feat      = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->feat_self = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->feat_nbrs = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->feat_rxcA = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->tmp_self  = XMALLOC(N_LAYERS_EXT * sizeof(VEC *));
    self->tmp_nbrs  = XMALLOC(N_LAYERS_EXT * sizeof(VEC *));

    ITER_LAYERS(l) {
        self->feat[l]      = MAT_alloc(N_VERTICES, DIM[l]);
        self->feat_self[l] = MAT_alloc(N_VERTICES, DIM[l]);
        self->feat_nbrs[l] = MAT_alloc(N_VERTICES, DIM[l]);
        if (l == 1)
            continue;
        self->feat_rxcA[l] = MAT_alloc(N_VERTICES, DIM[l]);
    }
    ITER_LAYERS_EXT(l) {
        self->tmp_self[l] = VEC_alloc(DIM[l]);
        self->tmp_nbrs[l] = VEC_alloc(DIM[l]);
    }
    self->pool = VEC_alloc(DIM_LAST);

    self->dirty_self      = BArray_alloc(N_VERTICES);
    self->dirty_self_next = BArray_alloc(N_VERTICES);
    self->dirty_nbrs      = BArray_alloc(N_VERTICES);
    self->dirty_nbrs_next = BArray_alloc(N_VERTICES);

    self->stack =
        Stack_alloc(sizeof(FManagerSnap), N_LAYERS, FManager_snap_free);
    self->stack_pool =
        Stack_alloc(sizeof(FManagerPoolSnap), 1, FManager_poolsnap_free);

    self->clock = 0;

    // init
    self->init         = true;
    self->feat[0]      = INPUT_FEAT;
    self->feat_rxcA[1] = INPUT_FEAT_RXCA;
    FManager_comp(self);
}

void FManager_push(FManager *self) {
    Stack_push_marker(self->stack);

    if (IS_GRAPH_CLASS) {
        FManagerPoolSnap *snap =
            (FManagerPoolSnap *)Stack_push(self->stack_pool, 1);
        if (Stack_fresh(self->stack_pool))
            snap->pool = VEC_alloc(DIM_LAST);
        VEC_copy(snap->pool, self->pool);
    }
}

void FManager_pop(FManager *self) {
    FManagerSnap *snap;
    while ((snap = (FManagerSnap *)Stack_pop(self->stack)) != NULL) {
        size_t l = snap->l;
        size_t v = snap->v;
        MV_SWAP(snap->feat_v, self->feat[l], v);
        if (snap->push_self)
            MV_SWAP(snap->feat_self_v, self->feat_self[l], v);
        if (snap->push_nbrs)
            MV_SWAP(snap->feat_nbrs_v, self->feat_nbrs[l], v);
        if (NOT_LAST_LAYER(l))
            MV_SWAP(snap->feat_rxcA_v, self->feat_rxcA[l + 1], v);
    }

    if (IS_GRAPH_CLASS) {
        FManagerPoolSnap *snap =
            (FManagerPoolSnap *)Stack_pop(self->stack_pool);
        VEC *tmp   = snap->pool;
        snap->pool = self->pool;
        self->pool = tmp;
    }
}

// ------------------------------------ sat ------------------------------------
bool FManager_sat(FManager *self) {
    PROFILING_START;

    if (self->init)
        self->init = false;
    else {
        if (USE_INC_COMP)
            FManager_update(self);
        else
            FManager_comp(self);
    }

    bool ret = false;
    if (IS_NODE_CLASS)
        ret = VEC_any_pos(MSLICE(self->feat[N_LAYERS], 0));
    if (IS_GRAPH_CLASS)
        ret = VEC_any_pos(self->pool);

    PROFILING_END;

    return ret;
}
