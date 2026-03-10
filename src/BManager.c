#include <assert.h>
#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "BManager.h"
#include "Common.h"
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

    MAT *lb;
    MAT *ub;
    MAT *lb_self;
    MAT *ub_self;
    MAT *lb_nbrs;
    MAT *ub_nbrs;
    MAT *lb_rxcA;
    MAT *ub_rxcA;

    VEC ub_pool;
} BManagerSnap;

static void BManager_snap_free(Node *node) {
    BManagerSnap *snap = (BManagerSnap *)node;

    ITER_LAYERS(l) {
        free(snap->index[l]);

        MAT_free(snap->lb[l]);
        MAT_free(snap->ub[l]);
        MAT_free(snap->lb_nbrs[l]);
        MAT_free(snap->ub_nbrs[l]);
        MAT_free(snap->lb_self[l]);
        MAT_free(snap->ub_self[l]);
        MAT_free(snap->lb_rxcA[l]);
        MAT_free(snap->ub_rxcA[l]);
    }
    VEC_free(snap->ub_pool);

    free(snap->index);
    free(snap->counter_self);
    free(snap->counter_nbrs);

    free(snap->lb);
    free(snap->ub);
    free(snap->lb_self);
    free(snap->ub_self);
    free(snap->lb_nbrs);
    free(snap->ub_nbrs);
    free(snap->lb_rxcA);
    free(snap->ub_rxcA);

    free(node);
}

// ----------------------------------- type ------------------------------------
struct BManager {
    GManager *gm;

    MAT *lb;
    MAT *ub;
    MAT *lb_self;
    MAT *ub_self;
    MAT *lb_nbrs;
    MAT *ub_nbrs;
    MAT *lb_rxcA;
    MAT *ub_rxcA;
    MAT *lb_tmp;
    MAT *ub_tmp;
    VEC  ub_pool;

    MAT *mtmp1;
    MAT *mtmp2;
    VEC *vtmp1;
    VEC *vtmp2;

    double **transpose;

    Stack         stack;
    BManagerSnap *snap;

    bool flush;

    bool    profiling;
    clock_t clock;
};

// --------------------------------- lifecycle ---------------------------------
BManager *BManager_alloc(GManager *gm) {
    BManager *self = XMALLOC(sizeof(BManager));

    self->gm = gm;

    // allocate
    self->lb      = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->ub      = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->lb_self = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->ub_self = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->lb_nbrs = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->ub_nbrs = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->lb_rxcA = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->ub_rxcA = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->lb_tmp  = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->ub_tmp  = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->vtmp1   = XMALLOC(N_LAYERS_EXT * sizeof(VEC *));
    self->vtmp2   = XMALLOC(N_LAYERS_EXT * sizeof(VEC *));
    self->mtmp1   = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->mtmp2   = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));

    ITER_LAYERS(l) {
        self->lb[l]      = MAT_alloc(N_VERTICES, DIM[l]);
        self->ub[l]      = MAT_alloc(N_VERTICES, DIM[l]);
        self->lb_nbrs[l] = MAT_alloc(N_VERTICES, DIM[l]);
        self->ub_nbrs[l] = MAT_alloc(N_VERTICES, DIM[l]);
        if (l == 1)
            continue;
        self->lb_self[l] = MAT_alloc(N_VERTICES, DIM[l]);
        self->ub_self[l] = MAT_alloc(N_VERTICES, DIM[l]);
        self->lb_rxcA[l] = MAT_alloc(N_VERTICES, DIM[l]);
        self->ub_rxcA[l] = MAT_alloc(N_VERTICES, DIM[l]);
    }
    ITER_LAYERS_EXT(l) {
        self->lb_tmp[l] = MAT_alloc(N_VERTICES, DIM[l]);
        self->ub_tmp[l] = MAT_alloc(N_VERTICES, DIM[l]);

        self->vtmp1[l] = VEC_alloc(DIM[l]);
        self->vtmp2[l] = VEC_alloc(DIM[l]);
        self->mtmp1[l] = MAT_alloc(N_VERTICES, DIM[l]);
        self->mtmp2[l] = MAT_alloc(N_VERTICES, DIM[l]);
    }
    self->ub_pool = VEC_alloc(DIM_LAST);

    self->transpose = XMALLOC(DIM_MAX * sizeof(double *));
    for (size_t i = 0; i < DIM_MAX; ++i)
        self->transpose[i] = XMALLOC(N_VERTICES * sizeof(double));

    Stack_init(&self->stack, sizeof(BManagerSnap));

    self->clock = 0;

    // init
    self->flush      = true;
    self->lb[0]      = INPUT_FEAT;
    self->ub[0]      = INPUT_FEAT;
    self->lb_self[1] = INPUT_FEAT_RXCC_CB;
    self->ub_self[1] = INPUT_FEAT_RXCC_CB;
    self->lb_rxcA[1] = INPUT_FEAT_RXCA;
    self->ub_rxcA[1] = INPUT_FEAT_RXCA;

    return self;
}

void BManager_free(BManager *self) {
    ITER_LAYERS(l) {
        MAT_free(self->lb[l]);
        MAT_free(self->ub[l]);
        MAT_free(self->lb_nbrs[l]);
        MAT_free(self->ub_nbrs[l]);
        if (l == 1)
            continue;
        MAT_free(self->lb_self[l]);
        MAT_free(self->ub_self[l]);
        MAT_free(self->lb_rxcA[l]);
        MAT_free(self->ub_rxcA[l]);
    }
    ITER_LAYERS_EXT(l) {
        MAT_free(self->lb_tmp[l]);
        MAT_free(self->ub_tmp[l]);

        MAT_free(self->mtmp1[l]);
        MAT_free(self->mtmp2[l]);
        VEC_free(self->vtmp1[l]);
        VEC_free(self->vtmp2[l]);
    }
    VEC_free(self->ub_pool);

    free(self->lb);
    free(self->ub);
    free(self->lb_self);
    free(self->ub_self);
    free(self->lb_nbrs);
    free(self->ub_nbrs);
    free(self->lb_rxcA);
    free(self->ub_rxcA);
    free(self->lb_tmp);
    free(self->ub_tmp);
    free(self->mtmp1);
    free(self->mtmp2);
    free(self->vtmp1);
    free(self->vtmp2);

    for (size_t i = 0; i < DIM_MAX; ++i)
        free(self->transpose[i]);
    free(self->transpose);

    Stack_cleanup(&self->stack, BManager_snap_free);

    free(self);
}
// ---------------------------------- getter -----------------------------------
MAT BManager_lb(BManager *self, size_t l) {
    return self->lb[l];
}

MAT BManager_ub(BManager *self, size_t l) {
    return self->ub[l];
}

// -------------------------------- statistics ---------------------------------
void BManager_set_profiling(BManager *self, bool profiling) {
    self->profiling = profiling;
}

clock_t BManager_clock(BManager *self) {
    return self->clock;
}

// ----------------------------------- snaps -----------------------------------
static void BManager_swap_snap(BManager *self, BManagerSnap *snap) {
    uint16_t **index        = snap->index;
    uint16_t  *counter_self = snap->counter_self;
    uint16_t  *counter_nbrs = snap->counter_nbrs;

    ITER_LAYERS(l) {
        uint16_t *index_l        = index[IS_NODE_CLASS ? l : 1];
        uint16_t  counter_self_l = counter_self[l];
        uint16_t  counter_nbrs_l = counter_nbrs[l];

        MAT lb_self      = self->lb_self[l];
        MAT ub_self      = self->ub_self[l];
        MAT lb_self_snap = snap->lb_self[l];
        MAT ub_self_snap = snap->ub_self[l];

        MAT lb_nbrs      = self->lb_nbrs[l];
        MAT ub_nbrs      = self->ub_nbrs[l];
        MAT lb_nbrs_snap = snap->lb_nbrs[l];
        MAT ub_nbrs_snap = snap->ub_nbrs[l];

        MAT lb      = self->lb[l];
        MAT ub      = self->ub[l];
        MAT lb_snap = snap->lb[l];
        MAT ub_snap = snap->ub[l];

        MAT lb_rxcA      = self->lb_rxcA[l + 1];
        MAT ub_rxcA      = self->ub_rxcA[l + 1];
        MAT lb_rxcA_snap = snap->lb_rxcA[l + 1];
        MAT ub_rxcA_snap = snap->ub_rxcA[l + 1];

        for (size_t i = 0; i < counter_self_l; ++i) {
            size_t v = index_l[i];

            VEC lb_self_tmp = lb_self[v];
            VEC ub_self_tmp = ub_self[v];

            lb_self[v] = lb_self_snap[i];
            ub_self[v] = ub_self_snap[i];

            lb_self_snap[i] = lb_self_tmp;
            ub_self_snap[i] = ub_self_tmp;
        }
        for (size_t i = 0; i < counter_nbrs_l; ++i) {
            size_t v = index_l[i];

            VEC lb_nbrs_tmp = lb_nbrs[v];
            VEC ub_nbrs_tmp = ub_nbrs[v];
            VEC lb_tmp      = lb[v];
            VEC ub_tmp      = ub[v];

            lb_nbrs[v] = lb_nbrs_snap[i];
            ub_nbrs[v] = ub_nbrs_snap[i];
            lb[v]      = lb_snap[i];
            ub[v]      = ub_snap[i];

            lb_nbrs_snap[i] = lb_nbrs_tmp;
            ub_nbrs_snap[i] = ub_nbrs_tmp;
            lb_snap[i]      = lb_tmp;
            ub_snap[i]      = ub_tmp;
        }
        if (NOT_LAST_LAYER(l)) {
            for (size_t i = 0; i < counter_nbrs_l; ++i) {
                size_t v = index_l[i];

                VEC lb_rxcA_tmp = lb_rxcA[v];
                VEC ub_rxcA_tmp = ub_rxcA[v];

                lb_rxcA[v] = lb_rxcA_snap[i];
                ub_rxcA[v] = ub_rxcA_snap[i];

                lb_rxcA_snap[i] = lb_rxcA_tmp;
                ub_rxcA_snap[i] = ub_rxcA_tmp;
            }
        }
    }
}

static void BManager_push_snap(BManager *self) {
    BManagerSnap *snap = self->snap;
    snap->ready        = true;

    // pool
    if (IS_GRAPH_CLASS) {
        VEC_copy(snap->ub_pool, self->ub_pool);

        uint16_t *index_l        = snap->index[IS_NODE_CLASS ? N_LAYERS : 1];
        uint16_t  counter_nbrs_l = snap->counter_nbrs[N_LAYERS];
        for (size_t i = 0; i < counter_nbrs_l; ++i) {
            size_t v = index_l[i];
            VV_sub(self->ub_pool, self->ub_pool, self->ub[N_LAYERS][v]);
        }
    }

    BManager_swap_snap(self, snap);
}

static void BManager_pop_snap(BManager *self) {
    BManagerSnap *snap = (BManagerSnap *)Stack_pop(&self->stack);
    if (!snap->ready)
        return;

    // pool
    if (IS_GRAPH_CLASS)
        VV_SWAP(self->ub_pool, snap->ub_pool);

    BManager_swap_snap(self, snap);
}

// --------------------------------- transpose ---------------------------------
static size_t BManager_transpose_signed(BManager *self, MAT feat, size_t v) {
    if (IS_DEL_ONLY) {
        ITER_MAT2(feat, i) {
            double *ptr = self->transpose[i];
            ITER_PINBRS(self->gm, v, u) {
                *ptr = -feat[u][i];
                ++ptr;
            }
        }

        return GManager_pideg(self->gm, v);
    }
    else {
        assert(IS_DEL_INS);
        ITER_MAT2(feat, i) {
            double *ptr = self->transpose[i];
            ITER_PINBRS(self->gm, v, u) {
                *ptr = -feat[u][i];
                ++ptr;
            }
            ITER_QINBRS(self->gm, v, u) {
                *ptr = feat[u][i];
                ++ptr;
            }
        }

        return GManager_pideg(self->gm, v) + GManager_qideg(self->gm, v);
    }

    return 0;
}

static size_t BManager_transpose_unsigned_p(BManager *self, MAT feat,
                                            size_t v) {
    ITER_MAT2(feat, i) {
        double *ptr = self->transpose[i];
        ITER_PINBRS(self->gm, v, u) {
            *ptr = feat[u][i];
            ++ptr;
        }
    }

    return GManager_pideg(self->gm, v);
}

static size_t BManager_transpose_unsigned(BManager *self, MAT feat, size_t v) {
    if (IS_DEL_ONLY) {
        ITER_MAT2(feat, i) {
            double *ptr = self->transpose[i];
            ITER_PINBRS(self->gm, v, u) {
                *ptr = feat[u][i];
                ++ptr;
            }
        }

        return GManager_pideg(self->gm, v);
    }
    else {
        assert(IS_DEL_INS);
        ITER_MAT2(feat, i) {
            double *ptr = self->transpose[i];
            ITER_PINBRS(self->gm, v, u) {
                *ptr = -feat[u][i];
                ++ptr;
            }
            ITER_QINBRS(self->gm, v, u) {
                *ptr = feat[u][i];
                ++ptr;
            }
        }

        return GManager_pideg(self->gm, v) + GManager_qideg(self->gm, v);
    }

    return 0;
}

// ------------------------------ comp bounds sum ------------------------------
static void BManager_nbrs_sum_fst(BManager *self, size_t v) {
    VEC lb_nbrs_v, ub_nbrs_v, tmp1, tmp2;
    MAT feat_p;
    if (USE_REORDER_COMP) {
        lb_nbrs_v = self->lb_nbrs[1][v];
        ub_nbrs_v = self->ub_nbrs[1][v];
        feat_p    = INPUT_FEAT_RXCA;
        tmp1      = self->vtmp1[1];
        tmp2      = self->vtmp2[1];
    }
    else {
        lb_nbrs_v = self->lb_tmp[0][v];
        ub_nbrs_v = self->ub_tmp[0][v];
        feat_p    = INPUT_FEAT;
        tmp1      = self->vtmp1[0];
        tmp2      = self->vtmp2[0];
    }

    uint16_t *plist   = GManager_pilist(self->gm, v);
    uint16_t *qlist   = GManager_qilist(self->gm, v);
    uint16_t *nlist   = GManager_nilist(self->gm, v);
    size_t    pdeg    = GManager_pideg(self->gm, v);
    size_t    qdeg    = GManager_qideg(self->gm, v);
    size_t    ndeg    = GManager_nideg(self->gm, v);
    size_t    lbudget = GManager_lbudget(self->gm, v);

    // for the input feat, lb = ub
    // normal
    MV_acc(lb_nbrs_v, NULL, feat_p, nlist, ndeg);
    VEC_copy(ub_nbrs_v, lb_nbrs_v);

    // potential
    if (pdeg + qdeg == 0) {  // do nothing
    }
    else if (lbudget >= pdeg + qdeg) {  // sum max/min of all
        MV_acc_neg(lb_nbrs_v, lb_nbrs_v, feat_p, plist, pdeg);
        MV_acc_pos(ub_nbrs_v, ub_nbrs_v, feat_p, plist, pdeg);
        if (IS_DEL_INS) {
            MV_acc_neg(lb_nbrs_v, lb_nbrs_v, feat_p, qlist, qdeg);
            MV_acc_pos(ub_nbrs_v, ub_nbrs_v, feat_p, qlist, qdeg);
        }
    }
    else if (lbudget == 0) {  // sum
        MV_acc(lb_nbrs_v, lb_nbrs_v, feat_p, plist, pdeg);
        VEC_copy(ub_nbrs_v, lb_nbrs_v);
    }
    else if ((lbudget == 1) && (qdeg == 0)) {
        // sum except the max/min of the min/max element
        VEC vmax = tmp1;
        VEC vmin = tmp2;

        MV_trinity(lb_nbrs_v, vmax, vmin, feat_p, plist, pdeg);
        VEC_copy(ub_nbrs_v, lb_nbrs_v);
        VV_sub_pos(lb_nbrs_v, vmax);
        VV_sub_neg(ub_nbrs_v, vmin);
    }
    else if ((lbudget == 2) && (qdeg == 0)) {
        // sum except the max/min of the top/last two elements
        MV_acc(lb_nbrs_v, lb_nbrs_v, feat_p, plist, pdeg);
        VEC_copy(ub_nbrs_v, lb_nbrs_v);

        MV_max2(tmp1, tmp2, feat_p, plist, pdeg);
        VV_sub_pos(lb_nbrs_v, tmp1);
        VV_sub_pos(lb_nbrs_v, tmp2);

        MV_min2(tmp1, tmp2, feat_p, plist, pdeg);
        VV_sub_neg(ub_nbrs_v, tmp1);
        VV_sub_neg(ub_nbrs_v, tmp2);
    }
    else {  // general case, sort and sum
        MV_acc(lb_nbrs_v, lb_nbrs_v, feat_p, plist, pdeg);
        VEC_copy(ub_nbrs_v, lb_nbrs_v);

        size_t counter = BManager_transpose_signed(self, feat_p, v);

        ITER_VEC(lb_nbrs_v, i) {  // lb
            double *ptr = self->transpose[i];
            qselect(ptr, counter, lbudget);
            for (size_t j = 0; j < lbudget; ++j)
                lb_nbrs_v[i] += MIN(ptr[j], 0.);
        }

        ITER_VEC(ub_nbrs_v, i) {  // ub
            double *ptr = self->transpose[i];
            qselect(ptr, counter, counter - lbudget);
            for (size_t j = 0; j < lbudget; ++j)
                ub_nbrs_v[i] += MAX(ptr[counter - lbudget + j], 0.);
        }
    }
}

static void BManager_nbrs_sum_tail(BManager *self, size_t l, size_t v) {
    VEC lb_nbrs_v, ub_nbrs_v, tmp1, tmp2;
    MAT lb_p, ub_p;
    if (USE_REORDER_COMP) {
        lb_nbrs_v = self->lb_nbrs[l][v];
        ub_nbrs_v = self->ub_nbrs[l][v];
        lb_p      = self->lb_rxcA[l];
        ub_p      = self->ub_rxcA[l];
        tmp1      = self->vtmp1[l];
        tmp2      = self->vtmp2[l];
    }
    else {
        lb_nbrs_v = self->lb_tmp[l - 1][v];
        ub_nbrs_v = self->ub_tmp[l - 1][v];
        lb_p      = self->lb[l - 1];
        ub_p      = self->ub[l - 1];
        tmp1      = self->vtmp1[l - 1];
        tmp2      = self->vtmp2[l - 1];
    }

    uint16_t *plist   = GManager_pilist(self->gm, v);
    uint16_t *qlist   = GManager_qilist(self->gm, v);
    uint16_t *nlist   = GManager_nilist(self->gm, v);
    size_t    pdeg    = GManager_pideg(self->gm, v);
    size_t    qdeg    = GManager_qideg(self->gm, v);
    size_t    ndeg    = GManager_nideg(self->gm, v);
    size_t    lbudget = GManager_lbudget(self->gm, v);

    // normal
    MV_acc(lb_nbrs_v, NULL, lb_p, nlist, ndeg);
    MV_acc(ub_nbrs_v, NULL, ub_p, nlist, ndeg);

    // potential
    if (pdeg + qdeg == 0) {  // do nothing
    }
    else if (lbudget >= pdeg + qdeg) {  // sum max/min of all
        MV_acc_neg(lb_nbrs_v, lb_nbrs_v, lb_p, plist, pdeg);
        MV_acc_pos(ub_nbrs_v, ub_nbrs_v, ub_p, plist, pdeg);
        if (IS_DEL_INS) {
            MV_acc_neg(lb_nbrs_v, lb_nbrs_v, lb_p, qlist, qdeg);
            MV_acc_pos(ub_nbrs_v, ub_nbrs_v, ub_p, qlist, qdeg);
        }
    }
    else if (lbudget == 0) {  // sum
        MV_acc(lb_nbrs_v, lb_nbrs_v, lb_p, plist, pdeg);
        MV_acc(ub_nbrs_v, ub_nbrs_v, ub_p, plist, pdeg);
    }
    else if ((lbudget == 1) && (qdeg == 0)) {
        // sum except the max/min of the min/max element
        MV_acc(lb_nbrs_v, lb_nbrs_v, lb_p, plist, pdeg);
        MV_acc(ub_nbrs_v, ub_nbrs_v, ub_p, plist, pdeg);

        MV_max(tmp1, NULL, lb_p, plist, pdeg);
        VV_sub_pos(lb_nbrs_v, tmp1);

        MV_min(tmp1, NULL, ub_p, plist, pdeg);
        VV_sub_neg(ub_nbrs_v, tmp1);
    }
    else if ((lbudget == 2) && (qdeg == 0)) {
        // sum except the max/min of the top/last two elements
        MV_acc(lb_nbrs_v, lb_nbrs_v, lb_p, plist, pdeg);
        MV_acc(ub_nbrs_v, ub_nbrs_v, ub_p, plist, pdeg);

        MV_max2(tmp1, tmp2, lb_p, plist, pdeg);
        VV_sub_pos(lb_nbrs_v, tmp1);
        VV_sub_pos(lb_nbrs_v, tmp2);

        MV_min2(tmp1, tmp2, ub_p, plist, pdeg);
        VV_sub_neg(ub_nbrs_v, tmp1);
        VV_sub_neg(ub_nbrs_v, tmp2);
    }
    else {  // general case, sort and sum
        // lb
        MV_acc(lb_nbrs_v, lb_nbrs_v, lb_p, plist, pdeg);
        size_t counter = BManager_transpose_signed(self, lb_p, v);
        ITER_VEC(lb_nbrs_v, i) {
            double *ptr = self->transpose[i];
            qselect(ptr, counter, lbudget);
            for (size_t j = 0; j < lbudget; ++j)
                lb_nbrs_v[i] += MIN(ptr[j], 0.);
        }

        // ub
        MV_acc(ub_nbrs_v, ub_nbrs_v, ub_p, plist, pdeg);
        counter = BManager_transpose_signed(self, ub_p, v);
        ITER_VEC(ub_nbrs_v, i) {
            double *ptr = self->transpose[i];
            qselect(ptr, counter, counter - lbudget);
            for (size_t j = 0; j < lbudget; ++j)
                ub_nbrs_v[i] += MAX(ptr[counter - lbudget + j], 0.);
        }
    }
}

static void BManager_nbrs_sum_loose(BManager *self, size_t l, size_t v) {
    VEC lb_nbrs_v, ub_nbrs_v;
    MAT lb_p, ub_p;
    if (USE_REORDER_COMP) {
        lb_nbrs_v = self->lb_nbrs[l][v];
        ub_nbrs_v = self->ub_nbrs[l][v];
        lb_p      = self->lb_rxcA[l];
        ub_p      = self->ub_rxcA[l];
    }
    else {
        lb_nbrs_v = self->lb_tmp[l - 1][v];
        ub_nbrs_v = self->ub_tmp[l - 1][v];
        lb_p      = self->lb[l - 1];
        ub_p      = self->ub[l - 1];
    }

    uint16_t *plist = GManager_pilist(self->gm, v);
    uint16_t *qlist = GManager_qilist(self->gm, v);
    uint16_t *nlist = GManager_nilist(self->gm, v);
    size_t    pdeg  = GManager_pideg(self->gm, v);
    size_t    qdeg  = GManager_qideg(self->gm, v);
    size_t    ndeg  = GManager_nideg(self->gm, v);

    // normal
    MV_acc(lb_nbrs_v, NULL, lb_p, nlist, ndeg);
    MV_acc(ub_nbrs_v, NULL, ub_p, nlist, ndeg);

    // potential
    MV_acc_neg(lb_nbrs_v, lb_nbrs_v, lb_p, plist, pdeg);
    MV_acc_pos(ub_nbrs_v, ub_nbrs_v, ub_p, plist, pdeg);
    if (IS_DEL_INS) {
        MV_acc_neg(lb_nbrs_v, lb_nbrs_v, lb_p, qlist, qdeg);
        MV_acc_pos(ub_nbrs_v, ub_nbrs_v, ub_p, qlist, qdeg);
    }
}

static void BManager_nbrs_sum(BManager *self, size_t l, size_t v) {
    if (USE_TIGHT_BOUND) {
        if (l == 1)
            BManager_nbrs_sum_fst(self, v);
        else
            BManager_nbrs_sum_tail(self, l, v);
    }
    else {
        BManager_nbrs_sum_loose(self, l, v);
    }
}

// ------------------------------ comp bounds max ------------------------------
static void BManager_nbrs_max_tight(BManager *self, size_t l, size_t v) {
    VEC lb_nbrs_v = self->lb_tmp[l - 1][v];
    VEC ub_nbrs_v = self->ub_tmp[l - 1][v];
    MAT lb_p      = self->lb[l - 1];
    MAT ub_p      = self->ub[l - 1];
    VEC tmp       = self->vtmp1[l - 1];
    VEC tmp2      = self->vtmp2[l - 1];

    uint16_t *plist   = GManager_pilist(self->gm, v);
    uint16_t *qlist   = GManager_qilist(self->gm, v);
    uint16_t *nlist   = GManager_nilist(self->gm, v);
    uint16_t *glist   = GManager_gilist(self->gm, v);
    size_t    pdeg    = GManager_pideg(self->gm, v);
    size_t    qdeg    = GManager_qideg(self->gm, v);
    size_t    ndeg    = GManager_nideg(self->gm, v);
    size_t    gdeg    = GManager_gideg(self->gm, v);
    size_t    lbudget = GManager_lbudget(self->gm, v);

    if (lbudget == 0) {
        if (gdeg == 0) {
            VEC_clear(lb_nbrs_v);
            VEC_clear(ub_nbrs_v);
        }
        else {
            MV_max(lb_nbrs_v, NULL, lb_p, plist, pdeg);
            MV_max(ub_nbrs_v, NULL, ub_p, plist, pdeg);
        }
    }
    else if ((ndeg == 0) && (pdeg == 0) && (qdeg == 0)) {
        VEC_clear(lb_nbrs_v);
        VEC_clear(ub_nbrs_v);
    }
    else {
        // lb
        if ((ndeg == 0) && (pdeg == 0)) {
            VEC_clear(lb_nbrs_v);
            if (qdeg > 0)
                MV_min(lb_nbrs_v, lb_nbrs_v, lb_p, qlist, qdeg);
        }
        else if ((ndeg > 0) && (pdeg == 0)) {
            MV_max(lb_nbrs_v, NULL, lb_p, nlist, ndeg);
        }
        else if ((ndeg == 0) && (pdeg > 0)) {
            if (lbudget >= pdeg) {
                VEC_clear(lb_nbrs_v);
                MV_min(lb_nbrs_v, lb_nbrs_v, lb_p, plist, pdeg);
                if (qdeg > 0)
                    MV_min(lb_nbrs_v, lb_nbrs_v, lb_p, qlist, qdeg);
            }
            else {
                if (lbudget == 1) {
                    MV_max2(tmp, lb_nbrs_v, lb_p, plist, pdeg);
                }
                else if (pdeg == lbudget + 1) {
                    MV_min(lb_nbrs_v, NULL, lb_p, plist, pdeg);
                }
                else if (pdeg == lbudget + 2) {
                    MV_min2(tmp, lb_nbrs_v, lb_p, plist, pdeg);
                }
                else if (pdeg == lbudget + 3) {
                    MV_min3(tmp, tmp2, lb_nbrs_v, lb_p, plist, pdeg);
                }
                else {
                    size_t counter =
                        BManager_transpose_unsigned_p(self, lb_p, v);
                    ITER_VEC(lb_nbrs_v, i) {
                        double *ptr = self->transpose[i];
                        qselect(ptr, counter, counter - lbudget - 1);
                        lb_nbrs_v[i] = ptr[counter - lbudget - 1];
                    }
                }
            }
        }
        else if ((ndeg > 0) && (pdeg > 0)) {
            if (lbudget >= pdeg) {
                MV_max(lb_nbrs_v, NULL, lb_p, nlist, ndeg);
            }
            else {
                if (lbudget == 1) {
                    MV_max2(tmp, lb_nbrs_v, lb_p, plist, pdeg);
                }
                else if (pdeg == lbudget + 1) {
                    MV_min(lb_nbrs_v, NULL, lb_p, plist, pdeg);
                }
                else if (pdeg == lbudget + 2) {
                    MV_min2(tmp, lb_nbrs_v, lb_p, plist, pdeg);
                }
                else if (pdeg == lbudget + 3) {
                    MV_min3(tmp, tmp2, lb_nbrs_v, lb_p, plist, pdeg);
                }
                else {
                    size_t counter =
                        BManager_transpose_unsigned_p(self, lb_p, v);
                    ITER_VEC(lb_nbrs_v, i) {
                        double *ptr = self->transpose[i];
                        qselect(ptr, counter, counter - lbudget - 1);
                        lb_nbrs_v[i] = ptr[counter - lbudget - 1];
                    }
                }
                MV_max(lb_nbrs_v, lb_nbrs_v, lb_p, nlist, ndeg);
            }
        }

        // ub
        if ((ndeg == 0) && (pdeg == 0)) {
            VEC_clear(ub_nbrs_v);
        }
        else if ((ndeg > 0) && (pdeg == 0)) {
            MV_max(ub_nbrs_v, NULL, ub_p, nlist, ndeg);
        }
        else if ((ndeg == 0) && (pdeg > 0)) {
            if (lbudget < pdeg) {
                MV_max(ub_nbrs_v, NULL, ub_p, plist, pdeg);
            }
            else {
                VEC_clear(ub_nbrs_v);
                MV_max(ub_nbrs_v, ub_nbrs_v, ub_p, plist, pdeg);
            }
        }
        else if ((ndeg > 0) && (pdeg > 0)) {
            MV_max(ub_nbrs_v, NULL, ub_p, glist, gdeg);
        }

        if (qdeg > 0)
            MV_max(ub_nbrs_v, ub_nbrs_v, ub_p, qlist, qdeg);
    }
}

static void BManager_nbrs_max_loose(BManager *self, size_t l, size_t v) {
    VEC lb_nbrs_v = self->lb_tmp[l - 1][v];
    VEC ub_nbrs_v = self->ub_tmp[l - 1][v];
    MAT lb_p      = self->lb[l - 1];
    MAT ub_p      = self->ub[l - 1];

    uint16_t *plist = GManager_pilist(self->gm, v);
    uint16_t *qlist = GManager_qilist(self->gm, v);
    uint16_t *nlist = GManager_nilist(self->gm, v);
    size_t    pdeg  = GManager_pideg(self->gm, v);
    size_t    qdeg  = GManager_qideg(self->gm, v);
    size_t    ndeg  = GManager_nideg(self->gm, v);

    if (ndeg > 0) {
        MV_max(lb_nbrs_v, NULL, lb_p, nlist, ndeg);
        MV_max(ub_nbrs_v, NULL, ub_p, nlist, ndeg);
        if (pdeg > 0)
            MV_max(ub_nbrs_v, ub_nbrs_v, ub_p, plist, pdeg);
        if (qdeg > 0)
            MV_max(ub_nbrs_v, ub_nbrs_v, ub_p, qlist, qdeg);
    }
    else {
        VEC_clear(lb_nbrs_v);
        VEC_clear(ub_nbrs_v);
        if (pdeg > 0) {
            MV_min(lb_nbrs_v, lb_nbrs_v, lb_p, plist, pdeg);
            MV_max(ub_nbrs_v, ub_nbrs_v, ub_p, plist, pdeg);
        }
        if (qdeg > 0) {
            MV_min(lb_nbrs_v, lb_nbrs_v, lb_p, qlist, qdeg);
            MV_max(ub_nbrs_v, ub_nbrs_v, ub_p, qlist, qdeg);
        }
    }
}

static void BManager_nbrs_max(BManager *self, size_t l, size_t v) {
    if (USE_TIGHT_BOUND)
        BManager_nbrs_max_tight(self, l, v);
    else
        BManager_nbrs_max_loose(self, l, v);
}

// ----------------------------- comp bounds mean ------------------------------
static void BManager_nbrs_mean_fst(BManager *self, size_t v) {
    VEC lb_nbrs_v, ub_nbrs_v;
    MAT feat_p;
    if (USE_REORDER_COMP) {
        lb_nbrs_v = self->lb_nbrs[1][v];
        ub_nbrs_v = self->ub_nbrs[1][v];
        feat_p    = INPUT_FEAT_RXCA;
    }
    else {
        lb_nbrs_v = self->lb_tmp[0][v];
        ub_nbrs_v = self->ub_tmp[0][v];
        feat_p    = INPUT_FEAT;
    }

    uint16_t *nlist   = GManager_nilist(self->gm, v);
    size_t    ndeg    = GManager_nideg(self->gm, v);
    size_t    lbudget = GManager_lbudget(self->gm, v);

    // normal
    MV_acc(lb_nbrs_v, NULL, feat_p, nlist, ndeg);
    MV_acc(ub_nbrs_v, NULL, feat_p, nlist, ndeg);

    // potential
    size_t counter = BManager_transpose_unsigned(self, feat_p, v);
    ITER_VEC(lb_nbrs_v, i) {
        double *ptr = self->transpose[i];
        qsortd(ptr, counter);

        // lb
        size_t n = ndeg;
        for (size_t j = 0; j < counter; ++j) {
            double val = ptr[j];
            if (n == 0 && val > 0.)
                break;
            if (!USE_TIGHT_BOUND || (j + lbudget >= counter))
                if (val * n > lb_nbrs_v[i])
                    break;
            lb_nbrs_v[i] += val;
            n++;
        }
        if (n > 0)
            lb_nbrs_v[i] = lb_nbrs_v[i] / n;

        // ub
        n = ndeg;
        for (size_t j = 0; j < counter; ++j) {
            double val = ptr[counter - j - 1];
            if (n == 0 && val < 0.)
                break;
            if (!USE_TIGHT_BOUND || (j + lbudget >= counter)) {
                if (val * n < ub_nbrs_v[i])
                    break;
            }
            ub_nbrs_v[i] += val;
            n++;
        }
        if (n > 0)
            ub_nbrs_v[i] = ub_nbrs_v[i] / n;
    }
}

static void BManager_nbrs_mean_tail(BManager *self, size_t l, size_t v) {
    VEC lb_nbrs_v, ub_nbrs_v;
    MAT lb_p, ub_p;
    if (USE_REORDER_COMP) {
        lb_nbrs_v = self->lb_nbrs[l][v];
        ub_nbrs_v = self->ub_nbrs[l][v];
        lb_p      = self->lb_rxcA[l];
        ub_p      = self->ub_rxcA[l];
    }
    else {
        lb_nbrs_v = self->lb_tmp[l - 1][v];
        ub_nbrs_v = self->ub_tmp[l - 1][v];
        lb_p      = self->lb[l - 1];
        ub_p      = self->ub[l - 1];
    }

    uint16_t *nlist   = GManager_nilist(self->gm, v);
    size_t    ndeg    = GManager_nideg(self->gm, v);
    size_t    lbudget = GManager_lbudget(self->gm, v);

    // normal
    MV_acc(lb_nbrs_v, NULL, lb_p, nlist, ndeg);
    MV_acc(ub_nbrs_v, NULL, ub_p, nlist, ndeg);

    // potential
    // lb
    size_t counter = BManager_transpose_unsigned(self, lb_p, v);
    ITER_VEC(lb_nbrs_v, i) {
        double *ptr = self->transpose[i];
        qsortd(ptr, counter);

        size_t n = ndeg;
        for (size_t j = 0; j < counter; ++j) {
            double val = ptr[j];
            if (n == 0 && val > 0.)
                break;
            if (!USE_TIGHT_BOUND || (j + lbudget >= counter)) {
                if (val * n > lb_nbrs_v[i])
                    break;
            }
            lb_nbrs_v[i] += val;
            ++n;
        }
        if (n > 0)
            lb_nbrs_v[i] = lb_nbrs_v[i] / n;
    }

    // ub
    counter = BManager_transpose_unsigned(self, ub_p, v);
    ITER_VEC(ub_nbrs_v, i) {
        double *ptr = self->transpose[i];
        qsortd(ptr, counter);

        size_t n = ndeg;
        for (size_t j = 0; j < counter; ++j) {
            double val = ptr[counter - j - 1];
            if (n == 0 && val < 0.)
                break;
            if (!USE_TIGHT_BOUND || (j + lbudget >= counter)) {
                if (val * n < ub_nbrs_v[i])
                    break;
            }
            ub_nbrs_v[i] += val;
            ++n;
        }
        if (n > 0)
            ub_nbrs_v[i] = ub_nbrs_v[i] / n;
    }
}

static void BManager_nbrs_mean_delins(BManager *self, size_t l, size_t v) {
    VEC    lb_nbrs_v, ub_nbrs_v;
    MAT    lb_p, ub_p;
    size_t n;
    if (USE_REORDER_COMP) {
        lb_nbrs_v = self->lb_nbrs[l][v];
        ub_nbrs_v = self->ub_nbrs[l][v];
        lb_p      = self->lb_rxcA[l];
        ub_p      = self->ub_rxcA[l];
    }
    else {
        lb_nbrs_v = self->lb_tmp[l - 1][v];
        ub_nbrs_v = self->ub_tmp[l - 1][v];
        lb_p      = self->lb[l - 1];
        ub_p      = self->ub[l - 1];
    }

    uint16_t *nlist = GManager_nilist(self->gm, v);
    size_t    ndeg  = GManager_nideg(self->gm, v);

    // normal
    MV_acc(lb_nbrs_v, NULL, lb_p, nlist, ndeg);
    MV_acc(ub_nbrs_v, NULL, ub_p, nlist, ndeg);

    // potential
    // lb
    size_t counter = BManager_transpose_unsigned(self, lb_p, v);
    ITER_VEC(lb_nbrs_v, i) {
        double *ptr = self->transpose[i];
        qsortd(ptr, counter);

        n = ndeg;
        for (size_t j = 0; j < counter; ++j) {
            double val = ptr[j];
            if (n == 0 && val > 0.)
                break;
            if (val * n > lb_nbrs_v[i])
                break;
            lb_nbrs_v[i] += val;
            n++;
        }
        if (n > 0)
            lb_nbrs_v[i] = lb_nbrs_v[i] / n;
    }

    // ub
    counter = BManager_transpose_unsigned(self, ub_p, v);
    ITER_VEC(ub_nbrs_v, i) {
        double *ptr = self->transpose[i];
        qsortd(ptr, counter);

        n = ndeg;
        for (size_t j = 0; j < counter; ++j) {
            double val = ptr[counter - j - 1];
            if (n == 0 && val < 0.)
                break;
            if (val * n < ub_nbrs_v[i])
                break;
            ub_nbrs_v[i] += val;
            n++;
        }
        if (n > 0)
            ub_nbrs_v[i] = ub_nbrs_v[i] / n;
    }
}

static void BManager_nbrs_mean(BManager *self, size_t l, size_t v) {
    if (IS_DEL_ONLY) {
        if (l == 1)
            BManager_nbrs_mean_fst(self, v);
        else
            BManager_nbrs_mean_tail(self, l, v);
    }
    else {
        BManager_nbrs_mean_delins(self, l, v);
    }
}

// -------------------------------- comp bounds --------------------------------
static void BManager_comp(BManager *self) {
    ITER_LAYERS(l) {
        // comp self
        if (l > 1)
            MM_relax_trans(self->lb_self[l], self->ub_self[l], self->lb[l - 1],
                           self->ub[l - 1], CC[l], CB[l], self->mtmp1[l - 1],
                           self->mtmp2[l - 1], NULL, 0);

        // comp nbrs
        if (IS_SUM_GNN) {
            ITER_VTXS(v) {
                BManager_nbrs_sum(self, l, v);
            }
        }
        else if (IS_MAX_GNN) {
            ITER_VTXS(v) {
                BManager_nbrs_max(self, l, v);
            }
        }
        else {
            assert(IS_MEAN_GNN);
            ITER_VTXS(v) {
                BManager_nbrs_mean(self, l, v);
            }
        }
        if (!USE_REORDER_COMP || IS_MAX_GNN)
            MM_relax_trans(self->lb_nbrs[l], self->ub_nbrs[l],
                           self->lb_tmp[l - 1], self->ub_tmp[l - 1], CA[l],
                           NULL, self->mtmp1[l - 1], self->mtmp2[l - 1], NULL,
                           0);

        // comp feat
        if (NOT_LAST_LAYER(l)) {
            ITER_VTXS(v) {
                VV_add_relu(self->lb[l][v], self->lb_self[l][v],
                            self->lb_nbrs[l][v]);
                VV_add_relu(self->ub[l][v], self->ub_self[l][v],
                            self->ub_nbrs[l][v]);
            }
        }
        else {
            ITER_VTXS(v) {
                VV_add(self->lb[l][v], self->lb_self[l][v],
                       self->lb_nbrs[l][v]);
                VV_add(self->ub[l][v], self->ub_self[l][v],
                       self->ub_nbrs[l][v]);
            }
        }

        // comp rxcA
        if (USE_REORDER_COMP && !IS_MAX_GNN && NOT_LAST_LAYER(l)) {
            MM_relax_trans(self->lb_rxcA[l + 1], self->ub_rxcA[l + 1],
                           self->lb[l], self->ub[l], CA[l + 1], NULL,
                           self->mtmp1[l], self->mtmp2[l], NULL, 0);
        }
    }

    if (IS_GRAPH_CLASS)
        MV_acc(self->ub_pool, CBP, self->ub[N_LAYERS], NULL, 0);
}

// ------------------------------- update bounds -------------------------------
static void BManager_update(BManager *self) {
    BManagerSnap *snap         = self->snap;
    uint16_t    **index        = snap->index;
    uint16_t     *counter_self = snap->counter_self;
    uint16_t     *counter_nbrs = snap->counter_nbrs;

    // comp index
    GManager_comp_index_b(self->gm, index, counter_self, counter_nbrs);

    // push
    BManager_push_snap(self);

    ITER_LAYERS(l) {
        uint16_t *index_l        = index[IS_NODE_CLASS ? l : 1];
        uint16_t  counter_self_l = counter_self[l];
        uint16_t  counter_nbrs_l = counter_nbrs[l];

        // comp self
        if (counter_self[l] > 0)
            MM_relax_trans(self->lb_self[l], self->ub_self[l], self->lb[l - 1],
                           self->ub[l - 1], CC[l], CB[l], self->mtmp1[l - 1],
                           self->mtmp2[l - 1], index_l, counter_self_l);

        // comp nbrs
        if (IS_SUM_GNN) {
            for (size_t i = 0; i < counter_nbrs_l; ++i) {
                size_t v = index_l[i];
                BManager_nbrs_sum(self, l, v);
            }
        }
        else if (IS_MAX_GNN) {
            for (size_t i = 0; i < counter_nbrs_l; ++i) {
                size_t v = index_l[i];
                BManager_nbrs_max(self, l, v);
            }
        }
        else {
            assert(IS_MEAN_GNN);
            for (size_t i = 0; i < counter_nbrs_l; ++i) {
                size_t v = index_l[i];
                BManager_nbrs_mean(self, l, v);
            }
        }
        if ((!USE_REORDER_COMP || IS_MAX_GNN) && (counter_nbrs_l > 0))
            MM_relax_trans(self->lb_nbrs[l], self->ub_nbrs[l],
                           self->lb_tmp[l - 1], self->ub_tmp[l - 1], CA[l],
                           NULL, self->mtmp1[l - 1], self->mtmp2[l - 1],
                           index_l, counter_nbrs_l);

        // comp bound
        if (NOT_LAST_LAYER(l)) {
            for (size_t i = 0; i < counter_nbrs_l; ++i) {
                size_t v = index_l[i];
                VV_add_relu(self->lb[l][v], self->lb_self[l][v],
                            self->lb_nbrs[l][v]);
                VV_add_relu(self->ub[l][v], self->ub_self[l][v],
                            self->ub_nbrs[l][v]);
            }
        }
        else {
            for (size_t i = 0; i < counter_nbrs_l; ++i) {
                size_t v = index_l[i];
                VV_add(self->lb[l][v], self->lb_self[l][v],
                       self->lb_nbrs[l][v]);
                VV_add(self->ub[l][v], self->ub_self[l][v],
                       self->ub_nbrs[l][v]);
            }
        }

        // comp rxcA
        if (USE_REORDER_COMP && !IS_MAX_GNN && NOT_LAST_LAYER(l))
            MM_relax_trans(self->lb_rxcA[l + 1], self->ub_rxcA[l + 1],
                           self->lb[l], self->ub[l], CA[l + 1], NULL,
                           self->mtmp1[l], self->mtmp2[l], index_l,
                           counter_nbrs_l);
    }

    // pool
    if (IS_GRAPH_CLASS) {
        uint16_t *index_l        = index[IS_NODE_CLASS ? N_LAYERS : 1];
        uint16_t  counter_nbrs_l = counter_nbrs[N_LAYERS];
        for (size_t i = 0; i < counter_nbrs_l; ++i) {
            size_t v = index_l[i];
            VV_add(self->ub_pool, self->ub_pool, self->ub[N_LAYERS][v]);
        }
    }
}

// ---------------------------------- status -----------------------------------
void BManager_push(BManager *self) {
    BManagerSnap *snap = (BManagerSnap *)Stack_push(&self->stack);

    if (Stack_fresh(&self->stack)) {
        snap->index        = XMALLOC(N_LAYERS_EXT * sizeof(uint16_t *));
        snap->counter_self = XMALLOC(N_LAYERS_EXT * sizeof(uint16_t));
        snap->counter_nbrs = XMALLOC(N_LAYERS_EXT * sizeof(uint16_t));

        snap->lb      = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
        snap->ub      = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
        snap->lb_self = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
        snap->ub_self = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
        snap->lb_nbrs = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
        snap->ub_nbrs = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
        snap->lb_rxcA = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
        snap->ub_rxcA = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));

        ITER_LAYERS(l) {
            snap->index[l]   = XMALLOC(N_VERTICES * sizeof(uint16_t));
            snap->lb[l]      = MAT_alloc(N_VERTICES, DIM[l]);
            snap->ub[l]      = MAT_alloc(N_VERTICES, DIM[l]);
            snap->lb_nbrs[l] = MAT_alloc(N_VERTICES, DIM[l]);
            snap->ub_nbrs[l] = MAT_alloc(N_VERTICES, DIM[l]);
            snap->lb_self[l] = MAT_alloc(N_VERTICES, DIM[l]);
            snap->ub_self[l] = MAT_alloc(N_VERTICES, DIM[l]);
            snap->lb_rxcA[l] = MAT_alloc(N_VERTICES, DIM[l]);
            snap->ub_rxcA[l] = MAT_alloc(N_VERTICES, DIM[l]);
        }
        snap->ub_pool = VEC_alloc(DIM_LAST);
    }

    snap->ready = false;

    self->snap = snap;
}

void BManager_pop(BManager *self) {
    BManager_pop_snap(self);
}

void BManager_flush(BManager *self) {
    self->flush = true;
}

// ------------------------------------ sat ------------------------------------
bool BManager_unsat(BManager *self) {
    PROFILING_START;

    if (self->flush || !USE_INC_COMP) {
        self->flush = false;
        BManager_comp(self);
    }
    else {
        BManager_update(self);
    }

    bool ret = false;
    if (IS_NODE_CLASS)
        ret = !VEC_any_pos(self->ub[N_LAYERS][0]);
    else {
        assert(IS_GRAPH_CLASS);
        ret = !VEC_any_pos(self->ub_pool);
    }

    PROFILING_END;

    return ret;
}
