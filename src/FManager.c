#include <assert.h>
#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "Common.h"
#include "FManager.h"
#include "GManager.h"
#include "Matrix.h"
#include "Stack.h"
#include "utils.h"

// ----------------------------------- snap ------------------------------------
typedef struct {
    Node node;

    bool ready;

    uint16_t **index;
    uint16_t  *counter_self;
    uint16_t  *counter_nbrs;

    MAT *feat;
    MAT *feat_self;
    MAT *feat_nbrs;
    MAT *feat_rxcA;

    VEC pool;
} FManagerSnap;

static void FManager_snap_free(Node *node) {
    FManagerSnap *snap = (FManagerSnap *)node;

    ITER_LAYERS(l) {
        free(snap->index[l]);

        MAT_free(snap->feat[l]);
        MAT_free(snap->feat_nbrs[l]);
        MAT_free(snap->feat_self[l]);
        MAT_free(snap->feat_rxcA[l]);
    }
    VEC_free(snap->pool);

    free(snap->index);
    free(snap->counter_self);
    free(snap->counter_nbrs);

    free(snap->feat);
    free(snap->feat_self);
    free(snap->feat_nbrs);
    free(snap->feat_rxcA);

    free(node);
}

// ----------------------------------- type ------------------------------------
struct FManager {
    GManager *gm;

    MAT *feat;
    MAT *feat_self;
    MAT *feat_nbrs;
    MAT *feat_rxcA;
    MAT *feat_tmp;
    VEC  pool;

    Stack         stack;
    FManagerSnap *snap;

    bool flush;

    bool    profiling;
    clock_t clock;
};

// --------------------------------- lifecycle ---------------------------------
FManager *FManager_alloc(GManager *gm) {
    FManager *self = XMALLOC(sizeof(FManager));

    self->gm = gm;

    // allocate
    self->feat      = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->feat_self = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->feat_nbrs = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->feat_rxcA = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->feat_tmp  = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));

    ITER_LAYERS(l) {
        self->feat[l]      = MAT_alloc(N_VERTICES, DIM[l]);
        self->feat_nbrs[l] = MAT_alloc(N_VERTICES, DIM[l]);
        if (l == 1)
            continue;
        self->feat_self[l] = MAT_alloc(N_VERTICES, DIM[l]);
        self->feat_rxcA[l] = MAT_alloc(N_VERTICES, DIM[l]);
    }
    ITER_LAYERS_EXT(l) {
        self->feat_tmp[l] = MAT_alloc(N_VERTICES, DIM[l]);
    }
    self->pool = VEC_alloc(DIM_LAST);

    Stack_init(&self->stack, sizeof(FManagerSnap));

    self->clock = 0;

    // init
    self->flush        = true;
    self->feat[0]      = INPUT_FEAT;
    self->feat_self[1] = INPUT_FEAT_RXCC_CB;
    self->feat_rxcA[1] = INPUT_FEAT_RXCA;

    return self;
}

void FManager_free(FManager *self) {
    ITER_LAYERS(l) {
        MAT_free(self->feat[l]);
        MAT_free(self->feat_nbrs[l]);
        if (l == 1)
            continue;
        MAT_free(self->feat_self[l]);
        MAT_free(self->feat_rxcA[l]);
    }
    ITER_LAYERS_EXT(l) {
        MAT_free(self->feat_tmp[l]);
    }
    VEC_free(self->pool);

    free(self->feat);
    free(self->feat_self);
    free(self->feat_nbrs);
    free(self->feat_rxcA);
    free(self->feat_tmp);

    Stack_cleanup(&self->stack, FManager_snap_free);

    free(self);
}

// ---------------------------------- getter -----------------------------------
MAT FManager_feat(FManager *self, size_t l) {
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
static void FManager_swap_snap(FManager *self, FManagerSnap *snap) {
    uint16_t **index        = snap->index;
    uint16_t  *counter_self = snap->counter_self;
    uint16_t  *counter_nbrs = snap->counter_nbrs;

    ITER_LAYERS(l) {
        uint16_t *index_l        = index[IS_NODE_CLASS ? l : 1];
        uint16_t  counter_self_l = counter_self[l];
        uint16_t  counter_nbrs_l = counter_nbrs[l];

        MAT feat_self      = self->feat_self[l];
        MAT feat_self_snap = snap->feat_self[l];

        MAT feat_nbrs      = self->feat_nbrs[l];
        MAT feat_nbrs_snap = snap->feat_nbrs[l];

        MAT feat      = self->feat[l];
        MAT feat_snap = snap->feat[l];

        MAT feat_rxcA      = self->feat_rxcA[l + 1];
        MAT feat_rxcA_snap = snap->feat_rxcA[l + 1];

        for (size_t i = 0; i < counter_self_l; ++i) {
            size_t v = index_l[i];

            VEC feat_self_tmp = feat_self[v];
            feat_self[v]      = feat_self_snap[i];
            feat_self_snap[i] = feat_self_tmp;
        }
        if (NOT_LAST_LAYER(l)) {
            for (size_t i = 0; i < counter_nbrs_l; ++i) {
                size_t v = index_l[i];

                VEC feat_nbrs_tmp = feat_nbrs[v];
                VEC feat_tmp      = feat[v];
                VEC feat_rxcA_tmp = feat_rxcA[v];

                feat_nbrs[v] = feat_nbrs_snap[i];
                feat[v]      = feat_snap[i];
                feat_rxcA[v] = feat_rxcA_snap[i];

                feat_nbrs_snap[i] = feat_nbrs_tmp;
                feat_snap[i]      = feat_tmp;
                feat_rxcA_snap[i] = feat_rxcA_tmp;
            }
        }
        else {
            for (size_t i = 0; i < counter_nbrs_l; ++i) {
                size_t v = index_l[i];

                VEC feat_nbrs_tmp = feat_nbrs[v];
                VEC feat_tmp      = feat[v];

                feat_nbrs[v] = feat_nbrs_snap[i];
                feat[v]      = feat_snap[i];

                feat_nbrs_snap[i] = feat_nbrs_tmp;
                feat_snap[i]      = feat_tmp;
            }
        }
    }
}

static void FManager_push_snap(FManager *self) {
    FManagerSnap *snap = self->snap;
    snap->ready        = true;

    // pool
    if (IS_GRAPH_CLASS) {
        VEC_copy(snap->pool, self->pool);

        uint16_t *index_l        = snap->index[IS_NODE_CLASS ? N_LAYERS : 1];
        uint16_t  counter_nbrs_l = snap->counter_nbrs[N_LAYERS];
        for (size_t i = 0; i < counter_nbrs_l; ++i) {
            size_t v = index_l[i];
            VV_sub(self->pool, self->pool, self->feat[N_LAYERS][v]);
        }
    }

    FManager_swap_snap(self, snap);
}

static void FManager_pop_snap(FManager *self) {
    FManagerSnap *snap = (FManagerSnap *)Stack_pop(&self->stack);
    if (!snap->ready)
        return;

    // pool
    if (IS_GRAPH_CLASS)
        VV_SWAP(self->pool, snap->pool);

    FManager_swap_snap(self, snap);
}

// ------------------------------ comp feat nbrs -------------------------------
static void FManager_nbrs_sum(FManager *self, size_t l, size_t v) {
    VEC feat_nbrs_v;
    MAT feat_p;
    if (USE_REORDER_COMP) {
        feat_nbrs_v = self->feat_nbrs[l][v];
        feat_p      = self->feat_rxcA[l];
    }
    else {
        feat_nbrs_v = self->feat_tmp[l - 1][v];
        feat_p      = self->feat[l - 1];
    }

    uint16_t *list = GManager_gilist(self->gm, v);
    size_t    deg  = GManager_gideg(self->gm, v);

    if (deg == 0)
        VEC_clear(feat_nbrs_v);
    else
        MV_acc(feat_nbrs_v, NULL, feat_p, list, deg);
}

static void FManager_nbrs_max(FManager *self, size_t l, size_t v) {
    VEC feat_nbrs_v = self->feat_tmp[l - 1][v];
    MAT feat_p      = self->feat[l - 1];

    uint16_t *list = GManager_gilist(self->gm, v);
    size_t    deg  = GManager_gideg(self->gm, v);

    if (deg == 0)
        VEC_clear(feat_nbrs_v);
    else
        MV_max(feat_nbrs_v, NULL, feat_p, list, deg);
}

static void FManager_nbrs_mean(FManager *self, size_t l, size_t v) {
    VEC feat_nbrs_v;
    MAT feat_p;
    if (USE_REORDER_COMP) {
        feat_nbrs_v = self->feat_nbrs[l][v];
        feat_p      = self->feat_rxcA[l];
    }
    else {
        feat_nbrs_v = self->feat_tmp[l - 1][v];
        feat_p      = self->feat[l - 1];
    }

    size_t    deg  = GManager_gideg(self->gm, v);
    uint16_t *list = GManager_gilist(self->gm, v);

    if (deg == 0)
        VEC_clear(feat_nbrs_v);
    else {
        MV_acc(feat_nbrs_v, NULL, feat_p, list, deg);
        VEC_div(feat_nbrs_v, deg);
    }
}

// --------------------------------- comp feat ---------------------------------
static void FManager_comp(FManager *self) {
    ITER_LAYERS(l) {
        // comp self
        if (l > 1)
            MM_dot_trans(self->feat_self[l], self->feat[l - 1], CC[l], CB[l],
                         NULL, 0);

        // comp nbrs
        if (IS_SUM_GNN) {
            ITER_VTXS(v) {
                FManager_nbrs_sum(self, l, v);
            }
        }
        else if (IS_MAX_GNN) {
            ITER_VTXS(v) {
                FManager_nbrs_max(self, l, v);
            }
        }
        else {
            assert(IS_MEAN_GNN);
            ITER_VTXS(v) {
                FManager_nbrs_mean(self, l, v);
            }
        }
        if (!USE_REORDER_COMP || IS_MAX_GNN)
            MM_dot_trans(self->feat_nbrs[l], self->feat_tmp[l - 1], CA[l], NULL,
                         NULL, 0);

        // comp feat
        if (NOT_LAST_LAYER(l)) {
            ITER_VTXS(v) {
                VV_add_relu(self->feat[l][v], self->feat_self[l][v],
                            self->feat_nbrs[l][v]);
            }
        }
        else {
            ITER_VTXS(v) {
                VV_add(self->feat[l][v], self->feat_self[l][v],
                       self->feat_nbrs[l][v]);
            }
        }

        // comp rxcA
        if (USE_REORDER_COMP && !IS_MAX_GNN && NOT_LAST_LAYER(l))
            MM_dot_trans(self->feat_rxcA[l + 1], self->feat[l], CA[l + 1], NULL,
                         NULL, 0);
    }

    if (IS_GRAPH_CLASS)
        MV_acc(self->pool, CBP, self->feat[N_LAYERS], NULL, 0);
}

// -------------------------------- update feat --------------------------------
static void FManager_update(FManager *self) {
    FManagerSnap *snap         = self->snap;
    uint16_t    **index        = snap->index;
    uint16_t     *counter_self = snap->counter_self;
    uint16_t     *counter_nbrs = snap->counter_nbrs;

    // comp index
    GManager_comp_index_f(self->gm, index, counter_self, counter_nbrs);

    // push
    FManager_push_snap(self);

    ITER_LAYERS(l) {
        uint16_t *index_l        = index[IS_NODE_CLASS ? l : 1];
        uint16_t  counter_self_l = counter_self[l];
        uint16_t  counter_nbrs_l = counter_nbrs[l];

        // comp self
        if (counter_self[l] > 0)
            MM_dot_trans(self->feat_self[l], self->feat[l - 1], CC[l], CB[l],
                         index_l, counter_self_l);

        // comp nbrs
        if (IS_SUM_GNN) {
            for (size_t i = 0; i < counter_nbrs_l; ++i) {
                size_t v = index_l[i];
                FManager_nbrs_sum(self, l, v);
            }
        }
        else if (IS_MAX_GNN) {
            for (size_t i = 0; i < counter_nbrs_l; ++i) {
                size_t v = index_l[i];
                FManager_nbrs_max(self, l, v);
            }
        }
        else {
            assert(IS_MEAN_GNN);
            for (size_t i = 0; i < counter_nbrs_l; ++i) {
                size_t v = index_l[i];
                FManager_nbrs_mean(self, l, v);
            }
        }
        if ((!USE_REORDER_COMP || IS_MAX_GNN) && (counter_nbrs_l > 0))
            MM_dot_trans(self->feat_nbrs[l], self->feat_tmp[l - 1], CA[l], NULL,
                         index_l, counter_nbrs_l);

        // comp feat
        if (NOT_LAST_LAYER(l)) {
            for (size_t i = 0; i < counter_nbrs_l; ++i) {
                size_t v = index_l[i];
                VV_add_relu(self->feat[l][v], self->feat_self[l][v],
                            self->feat_nbrs[l][v]);
            }
        }
        else {
            for (size_t i = 0; i < counter_nbrs_l; ++i) {
                size_t v = index_l[i];
                VV_add(self->feat[l][v], self->feat_self[l][v],
                       self->feat_nbrs[l][v]);
            }
        }

        // comp rxcA
        if (USE_REORDER_COMP && !IS_MAX_GNN && NOT_LAST_LAYER(l))
            MM_dot_trans(self->feat_rxcA[l + 1], self->feat[l], CA[l + 1], NULL,
                         index_l, counter_nbrs_l);
    }

    // pool
    if (IS_GRAPH_CLASS) {
        uint16_t *index_l        = index[IS_NODE_CLASS ? N_LAYERS : 1];
        uint16_t  counter_nbrs_l = counter_nbrs[N_LAYERS];
        for (size_t i = 0; i < counter_nbrs_l; ++i) {
            size_t v = index_l[i];
            VV_add(self->pool, self->pool, self->feat[N_LAYERS][v]);
        }
    }
}

// ---------------------------------- status -----------------------------------
void FManager_push(FManager *self) {
    FManagerSnap *snap = (FManagerSnap *)Stack_push(&self->stack);

    if (Stack_fresh(&self->stack)) {
        snap->index        = XMALLOC(N_LAYERS_EXT * sizeof(uint16_t *));
        snap->counter_self = XMALLOC(N_LAYERS_EXT * sizeof(uint16_t));
        snap->counter_nbrs = XMALLOC(N_LAYERS_EXT * sizeof(uint16_t));

        snap->feat      = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
        snap->feat_self = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
        snap->feat_nbrs = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
        snap->feat_rxcA = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));

        ITER_LAYERS(l) {
            snap->index[l]     = XMALLOC(N_VERTICES * sizeof(uint16_t));
            snap->feat[l]      = MAT_alloc(N_VERTICES, DIM[l]);
            snap->feat_nbrs[l] = MAT_alloc(N_VERTICES, DIM[l]);
            snap->feat_self[l] = MAT_alloc(N_VERTICES, DIM[l]);
            snap->feat_rxcA[l] = MAT_alloc(N_VERTICES, DIM[l]);
        }
        snap->pool = VEC_alloc(DIM_LAST);
    }

    snap->ready = false;

    self->snap = snap;
}

void FManager_pop(FManager *self) {
    FManager_pop_snap(self);
}

// ------------------------------------ sat ------------------------------------
bool FManager_sat(FManager *self) {
    PROFILING_START;

    if (self->flush || !USE_INC_COMP) {
        self->flush = false;
        FManager_comp(self);
    }
    else {
        FManager_update(self);
    }

    bool ret = false;
    if (IS_NODE_CLASS)
        ret = VEC_any_pos(self->feat[N_LAYERS][0]);
    else {
        assert(IS_GRAPH_CLASS);
        ret = VEC_any_pos(self->pool);
    }

    PROFILING_END;

    return ret;
}
