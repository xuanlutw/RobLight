#include <assert.h>
#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "BArray.h"
#include "BManager.h"
#include "Common.h"
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

    VEC *lb_v;
    VEC *ub_v;
    VEC *lb_self_v;
    VEC *ub_self_v;
    VEC *lb_nbrs_v;
    VEC *ub_nbrs_v;
    VEC *lb_rxcA_v;
    VEC *ub_rxcA_v;
} BManagerSnap;

typedef struct {
    Node node;

    VEC *ub_pool;
} BManagerPoolSnap;

static void BManager_snap_free(Node *node) {
    BManagerSnap *snap = (BManagerSnap *)node;

    if (((Node *)snap)->type > 0) {
        VEC_free(snap->lb_v);
        VEC_free(snap->ub_v);
        VEC_free(snap->lb_self_v);
        VEC_free(snap->ub_self_v);
        VEC_free(snap->lb_nbrs_v);
        VEC_free(snap->ub_nbrs_v);
        if (snap->lb_rxcA_v != NULL) {
            VEC_free(snap->lb_rxcA_v);
            VEC_free(snap->ub_rxcA_v);
        }
    }

    free(snap);
}

static void BManager_poolsnap_free(Node *node) {
    BManagerPoolSnap *snap = (BManagerPoolSnap *)node;

    if (snap->ub_pool != NULL)
        VEC_free(snap->ub_pool);

    free(snap);
}

// ----------------------------------- type ------------------------------------
struct BManager {
    Common *common;
    PGraph *G;

    MAT **lb;
    MAT **ub;
    MAT **lb_self;
    MAT **ub_self;
    MAT **lb_nbrs;
    MAT **ub_nbrs;
    MAT **lb_rxcA;
    MAT **ub_rxcA;
    VEC **tmp_lb_self;
    VEC **tmp_ub_self;
    VEC **tmp_lb_nbrs;
    VEC **tmp_ub_nbrs;
    VEC  *ub_pool;

    VEC **tmp_vec1;
    VEC **tmp_vec2;

    double *tmp;

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
BManager *BManager_alloc(Common *common, PGraph *G) {
    BManager *self = XMALLOC(sizeof(BManager));

    self->common = common;
    self->G      = G;

    return self;
}

void BManager_free(BManager *self) {
    ITER_LAYERS(l) {
        MAT_free(self->lb[l]);
        MAT_free(self->ub[l]);
        MAT_free(self->lb_self[l]);
        MAT_free(self->ub_self[l]);
        MAT_free(self->lb_nbrs[l]);
        MAT_free(self->ub_nbrs[l]);
        if (l == 1)
            continue;
        MAT_free(self->lb_rxcA[l]);
        MAT_free(self->ub_rxcA[l]);
    }
    ITER_LAYERS_EXT(l) {
        VEC_free(self->tmp_lb_self[l]);
        VEC_free(self->tmp_ub_self[l]);
        VEC_free(self->tmp_lb_nbrs[l]);
        VEC_free(self->tmp_ub_nbrs[l]);
        VEC_free(self->tmp_vec1[l]);
        VEC_free(self->tmp_vec2[l]);
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
    free(self->tmp_lb_self);
    free(self->tmp_ub_self);
    free(self->tmp_lb_nbrs);
    free(self->tmp_ub_nbrs);
    free(self->tmp_vec1);
    free(self->tmp_vec2);

    free(self->tmp);

    BArray_free(self->dirty_self);
    BArray_free(self->dirty_self_next);
    BArray_free(self->dirty_nbrs);
    BArray_free(self->dirty_nbrs_next);

    Stack_free(self->stack);
    Stack_free(self->stack_pool);

    free(self);
}

// ---------------------------------- getter -----------------------------------
MAT *BManager_lb(BManager *self, size_t l) {
    return self->lb[l];
}

MAT *BManager_ub(BManager *self, size_t l) {
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
static void BManager_push_snap(BManager *self, size_t l, size_t v,
                               bool push_self, bool push_nbrs) {
    BManagerSnap *snap = (BManagerSnap *)Stack_push(self->stack, l);

    if (Stack_fresh(self->stack)) {
        snap->lb_v      = VEC_alloc(DIM[l]);
        snap->ub_v      = VEC_alloc(DIM[l]);
        snap->lb_self_v = VEC_alloc(DIM[l]);
        snap->ub_self_v = VEC_alloc(DIM[l]);
        snap->lb_nbrs_v = VEC_alloc(DIM[l]);
        snap->ub_nbrs_v = VEC_alloc(DIM[l]);
        if (NOT_LAST_LAYER(l)) {
            snap->lb_rxcA_v = VEC_alloc(DIM[l + 1]);
            snap->ub_rxcA_v = VEC_alloc(DIM[l + 1]);
        }
        else {
            snap->lb_rxcA_v = NULL;
            snap->ub_rxcA_v = NULL;
        }
    }

    snap->l         = l;
    snap->v         = v;
    snap->push_self = push_self;
    snap->push_nbrs = push_nbrs;
    MV_SWAP(snap->lb_v, self->lb[l], v);
    MV_SWAP(snap->ub_v, self->ub[l], v);
    if (push_self) {
        MV_SWAP(snap->lb_self_v, self->lb_self[l], v);
        MV_SWAP(snap->ub_self_v, self->ub_self[l], v);
    }
    if (push_nbrs) {
        MV_SWAP(snap->lb_nbrs_v, self->lb_nbrs[l], v);
        MV_SWAP(snap->ub_nbrs_v, self->ub_nbrs[l], v);
    }
    if (NOT_LAST_LAYER(l)) {
        MV_SWAP(snap->lb_rxcA_v, self->lb_rxcA[l + 1], v);
        MV_SWAP(snap->ub_rxcA_v, self->ub_rxcA[l + 1], v);
    }
}

// ------------------------------ comp bounds sum ------------------------------
static void BManager_comp_nbrs_sum_fst(BManager *self, size_t v) {
    VEC *lb_nbrs_v, *ub_nbrs_v, *tmp1, *tmp2;
    MAT *feat_p;
    if (USE_REORDER_COMP) {
        lb_nbrs_v = self->tmp_lb_nbrs[1];
        ub_nbrs_v = self->tmp_ub_nbrs[1];
        feat_p    = INPUT_FEAT_RXCA;
        tmp1      = self->tmp_vec1[1];
        tmp2      = self->tmp_vec2[1];
    }
    if (!USE_REORDER_COMP) {
        lb_nbrs_v = self->tmp_lb_nbrs[0];
        ub_nbrs_v = self->tmp_ub_nbrs[0];
        feat_p    = INPUT_FEAT;
        tmp1      = self->tmp_vec1[0];
        tmp2      = self->tmp_vec2[0];
    }

    // for the input feat, lb = ub
    // normal
    MV_acc(lb_nbrs_v, NULL, feat_p, PGraph_niadj(self->G, v));
    VEC_copy(ub_nbrs_v, lb_nbrs_v);

    // potential
    size_t pideg   = PGraph_pideg(self->G, v);
    size_t qideg   = IS_DEL_INS ? PGraph_qideg(self->G, v) : 0;
    size_t lbudget = PGraph_lbudget(self->G, v);
    // do nothing
    if (pideg + qideg == 0) {
    }
    // sum max/min of all
    else if (lbudget >= pideg + qideg) {
        MV_acc_neg(lb_nbrs_v, lb_nbrs_v, feat_p, PGraph_piadj(self->G, v));
        MV_acc_pos(ub_nbrs_v, ub_nbrs_v, feat_p, PGraph_piadj(self->G, v));
        if (IS_DEL_INS) {
            MV_acc_neg(lb_nbrs_v, lb_nbrs_v, feat_p, PGraph_qiadj(self->G, v));
            MV_acc_pos(ub_nbrs_v, ub_nbrs_v, feat_p, PGraph_qiadj(self->G, v));
        }
    }
    // sum
    else if (lbudget == 0) {
        MV_acc(lb_nbrs_v, lb_nbrs_v, feat_p, PGraph_piadj(self->G, v));
        VEC_copy(ub_nbrs_v, lb_nbrs_v);
    }
    // sum except the max/min of the min/max element
    else if ((lbudget == 1) && (qideg == 0)) {
        VEC *vmax = tmp1;
        VEC *vmin = tmp2;

        MV_trinity(lb_nbrs_v, vmax, vmin, feat_p, PGraph_piadj(self->G, v));
        VEC_copy(ub_nbrs_v, lb_nbrs_v);
        VV_sub_pos(lb_nbrs_v, vmax);
        VV_sub_neg(ub_nbrs_v, vmin);
    }
    // sum except the max/min of the top/last two elements
    else if ((lbudget == 2) && (qideg == 0)) {
        MV_acc(lb_nbrs_v, lb_nbrs_v, feat_p, PGraph_piadj(self->G, v));
        VEC_copy(ub_nbrs_v, lb_nbrs_v);

        MV_max2(tmp1, tmp2, feat_p, PGraph_piadj(self->G, v));
        VV_sub_pos(lb_nbrs_v, tmp1);
        VV_sub_pos(lb_nbrs_v, tmp2);

        MV_min2(tmp1, tmp2, feat_p, PGraph_piadj(self->G, v));
        VV_sub_neg(ub_nbrs_v, tmp1);
        VV_sub_neg(ub_nbrs_v, tmp2);
    }
    // general case, sort and sum
    else {
        MV_acc(lb_nbrs_v, lb_nbrs_v, feat_p, PGraph_piadj(self->G, v));
        VEC_copy(ub_nbrs_v, lb_nbrs_v);
        ITER_VEC(lb_nbrs_v, i) {
            size_t counter = 0;
            size_t j;
            ITER_PINBRS(self->G, v, u) {
                self->tmp[counter] = -MVAL(feat_p, u, i);
                ++counter;
            }
            if (IS_DEL_INS) {
                ITER_QINBRS(self->G, v, u) {
                    self->tmp[counter] = MVAL(feat_p, u, i);
                    ++counter;
                }
            }

            // lb
            qselect(self->tmp, counter, lbudget);
            for (j = 0; j < lbudget; ++j) {
                double val = self->tmp[j];
                VVAL(lb_nbrs_v, i) += MIN(val, 0.);
            }

            // ub
            qselect(self->tmp, counter, counter - lbudget);
            for (j = 0; j < lbudget; ++j) {
                double val = self->tmp[counter - lbudget + j];
                VVAL(ub_nbrs_v, i) += MAX(val, 0.);
            }
        }
    }

    if (!USE_REORDER_COMP)
        MV_relax(self->tmp_lb_nbrs[1], self->tmp_ub_nbrs[1], CA[1], CA_ABS[1],
                 lb_nbrs_v, ub_nbrs_v, self->tmp_vec1[0], self->tmp_vec2[0]);
}

static void BManager_comp_nbrs_sum_tail(BManager *self, size_t l, size_t v) {
    VEC *lb_nbrs_v, *ub_nbrs_v, *tmp1, *tmp2;
    MAT *lb_p, *ub_p;
    if (USE_REORDER_COMP) {
        lb_nbrs_v = self->tmp_lb_nbrs[l];
        ub_nbrs_v = self->tmp_ub_nbrs[l];
        lb_p      = self->lb_rxcA[l];
        ub_p      = self->ub_rxcA[l];
        tmp1      = self->tmp_vec1[l];
        tmp2      = self->tmp_vec2[l];
    }
    if (!USE_REORDER_COMP) {
        lb_nbrs_v = self->tmp_lb_nbrs[l - 1];
        ub_nbrs_v = self->tmp_ub_nbrs[l - 1];
        lb_p      = self->lb[l - 1];
        ub_p      = self->ub[l - 1];
        tmp1      = self->tmp_vec1[l - 1];
        tmp2      = self->tmp_vec2[l - 1];
    }

    // normal
    MV_acc(lb_nbrs_v, NULL, lb_p, PGraph_niadj(self->G, v));
    MV_acc(ub_nbrs_v, NULL, ub_p, PGraph_niadj(self->G, v));

    // potential
    size_t pideg   = PGraph_pideg(self->G, v);
    size_t qideg   = IS_DEL_INS ? PGraph_qideg(self->G, v) : 0;
    size_t lbudget = PGraph_lbudget(self->G, v);
    // do nothing
    if (pideg + qideg == 0) {
    }
    // sum max/min of all
    else if (lbudget >= pideg + qideg) {
        MV_acc_neg(lb_nbrs_v, lb_nbrs_v, lb_p, PGraph_piadj(self->G, v));
        MV_acc_pos(ub_nbrs_v, ub_nbrs_v, ub_p, PGraph_piadj(self->G, v));
        if (IS_DEL_INS) {
            MV_acc_neg(lb_nbrs_v, lb_nbrs_v, lb_p, PGraph_qiadj(self->G, v));
            MV_acc_pos(ub_nbrs_v, ub_nbrs_v, ub_p, PGraph_qiadj(self->G, v));
        }
    }
    // sum
    else if (lbudget == 0) {
        MV_acc(lb_nbrs_v, lb_nbrs_v, lb_p, PGraph_piadj(self->G, v));
        MV_acc(ub_nbrs_v, ub_nbrs_v, ub_p, PGraph_piadj(self->G, v));
    }
    // sum except the max/min of the min/max element
    else if ((lbudget == 1) && (qideg == 0)) {
        MV_acc(lb_nbrs_v, lb_nbrs_v, lb_p, PGraph_piadj(self->G, v));
        MV_acc(ub_nbrs_v, ub_nbrs_v, ub_p, PGraph_piadj(self->G, v));

        MV_max(tmp1, NULL, lb_p, PGraph_piadj(self->G, v));
        VV_sub_pos(lb_nbrs_v, tmp1);

        MV_min(tmp1, NULL, ub_p, PGraph_piadj(self->G, v));
        VV_sub_neg(ub_nbrs_v, tmp1);
    }
    // sum except the max/min of the top/last two elements
    else if ((lbudget == 2) && (qideg == 0)) {
        MV_acc(lb_nbrs_v, lb_nbrs_v, lb_p, PGraph_piadj(self->G, v));
        MV_acc(ub_nbrs_v, ub_nbrs_v, ub_p, PGraph_piadj(self->G, v));

        MV_max2(tmp1, tmp2, lb_p, PGraph_piadj(self->G, v));
        VV_sub_pos(lb_nbrs_v, tmp1);
        VV_sub_pos(lb_nbrs_v, tmp2);

        MV_min2(tmp1, tmp2, ub_p, PGraph_piadj(self->G, v));
        VV_sub_neg(ub_nbrs_v, tmp1);
        VV_sub_neg(ub_nbrs_v, tmp2);
    }
    // general case, sort and sum
    else {
        // lb
        MV_acc(lb_nbrs_v, lb_nbrs_v, lb_p, PGraph_piadj(self->G, v));
        ITER_VEC(lb_nbrs_v, i) {
            size_t counter = 0;
            size_t j;
            ITER_PINBRS(self->G, v, u) {
                self->tmp[counter] = -MVAL(lb_p, u, i);
                ++counter;
            }
            if (IS_DEL_INS) {
                ITER_QINBRS(self->G, v, u) {
                    self->tmp[counter] = MVAL(lb_p, u, i);
                    ++counter;
                }
            }
            qselect(self->tmp, counter, lbudget);
            for (j = 0; j < lbudget; ++j) {
                double val = self->tmp[j];
                VVAL(lb_nbrs_v, i) += MIN(val, 0.);
            }
        }

        // ub
        MV_acc(ub_nbrs_v, ub_nbrs_v, ub_p, PGraph_piadj(self->G, v));
        ITER_VEC(ub_nbrs_v, i) {
            size_t counter = 0;
            size_t j;
            ITER_PINBRS(self->G, v, u) {
                self->tmp[counter] = -MVAL(ub_p, u, i);
                ++counter;
            }
            if (IS_DEL_INS) {
                ITER_QINBRS(self->G, v, u) {
                    self->tmp[counter] = MVAL(ub_p, u, i);
                    ++counter;
                }
            }
            qselect(self->tmp, counter, counter - lbudget);
            for (j = 0; j < lbudget; ++j) {
                double val = self->tmp[counter - lbudget + j];
                VVAL(ub_nbrs_v, i) += MAX(val, 0.);
            }
        }
    }

    if (!USE_REORDER_COMP)
        MV_relax(self->tmp_lb_nbrs[l], self->tmp_ub_nbrs[l], CA[l], CA_ABS[l],
                 lb_nbrs_v, ub_nbrs_v, self->tmp_vec1[l - 1],
                 self->tmp_vec2[l - 1]);
}

static void BManager_comp_nbrs_sum_loose(BManager *self, size_t l, size_t v) {
    VEC *lb_nbrs_v, *ub_nbrs_v;
    MAT *lb_p, *ub_p;
    if (USE_REORDER_COMP) {
        lb_nbrs_v = self->tmp_lb_nbrs[l];
        ub_nbrs_v = self->tmp_ub_nbrs[l];
        lb_p      = self->lb_rxcA[l];
        ub_p      = self->ub_rxcA[l];
    }
    if (!USE_REORDER_COMP) {
        lb_nbrs_v = self->tmp_lb_nbrs[l - 1];
        ub_nbrs_v = self->tmp_ub_nbrs[l - 1];
        lb_p      = self->lb[l - 1];
        ub_p      = self->ub[l - 1];
    }

    // normal
    MV_acc(lb_nbrs_v, NULL, lb_p, PGraph_niadj(self->G, v));
    MV_acc(ub_nbrs_v, NULL, ub_p, PGraph_niadj(self->G, v));

    // potential deletion
    MV_acc_neg(lb_nbrs_v, lb_nbrs_v, lb_p, PGraph_piadj(self->G, v));
    MV_acc_pos(ub_nbrs_v, ub_nbrs_v, ub_p, PGraph_piadj(self->G, v));

    // potential insertion
    if (IS_DEL_INS) {
        MV_acc_neg(lb_nbrs_v, lb_nbrs_v, lb_p, PGraph_qiadj(self->G, v));
        MV_acc_pos(ub_nbrs_v, ub_nbrs_v, ub_p, PGraph_qiadj(self->G, v));
    }

    if (!USE_REORDER_COMP)
        MV_relax(self->tmp_lb_nbrs[l], self->tmp_ub_nbrs[l], CA[l], CA_ABS[l],
                 lb_nbrs_v, ub_nbrs_v, self->tmp_vec1[l - 1],
                 self->tmp_vec2[l - 1]);
}

static void BManager_comp_nbrs_sum(BManager *self, size_t l, size_t v) {
    if (USE_TIGHT_BOUND) {
        if (l == 1)
            BManager_comp_nbrs_sum_fst(self, v);
        else
            BManager_comp_nbrs_sum_tail(self, l, v);
    }
    else {
        BManager_comp_nbrs_sum_loose(self, l, v);
    }
}

// ------------------------------ comp bounds max ------------------------------
static void BManager_comp_nbrs_max_tight(BManager *self, size_t l, size_t v) {
    VEC *lb_nbrs_v = self->tmp_lb_nbrs[l - 1];
    VEC *ub_nbrs_v = self->tmp_ub_nbrs[l - 1];
    MAT *lb_p      = self->lb[l - 1];
    MAT *ub_p      = self->ub[l - 1];
    VEC *tmp       = self->tmp_vec1[l - 1];

    size_t lbudget = PGraph_lbudget(self->G, v);
    size_t nideg   = PGraph_nideg(self->G, v);
    size_t pideg   = PGraph_pideg(self->G, v);
    size_t qideg   = IS_DEL_INS ? PGraph_qideg(self->G, v) : 0;
    size_t ideg    = PGraph_ideg(self->G, v);
    if (lbudget == 0) {
        if (ideg == 0) {
            VEC_clear(self->tmp_lb_nbrs[l]);
            VEC_clear(self->tmp_ub_nbrs[l]);
            return;  // bypass relaxation
        }
        else {
            MV_max(lb_nbrs_v, NULL, lb_p, PGraph_iadj(self->G, v));
            MV_max(ub_nbrs_v, NULL, ub_p, PGraph_iadj(self->G, v));
        }
    }
    else if ((nideg == 0) && (pideg == 0) && (qideg == 0)) {
        VEC_clear(self->tmp_lb_nbrs[l]);
        VEC_clear(self->tmp_ub_nbrs[l]);
        return;  // bypass relaxation
    }
    else {
        // lb
        if ((nideg == 0) && (pideg == 0)) {
            VEC_clear(lb_nbrs_v);
            if (qideg > 0)
                MV_min(lb_nbrs_v, lb_nbrs_v, lb_p, PGraph_qiadj(self->G, v));
        }
        else if ((nideg > 0) && (pideg == 0)) {
            MV_max(lb_nbrs_v, NULL, lb_p, PGraph_niadj(self->G, v));
        }
        else if ((nideg == 0) && (pideg > 0)) {
            if (lbudget < pideg) {
                if (lbudget == 1) {
                    MV_max2(tmp, lb_nbrs_v, lb_p, PGraph_piadj(self->G, v));
                }
                else {
                    ITER_VEC(lb_nbrs_v, i) {
                        size_t counter = 0;
                        ITER_PINBRS(self->G, v, u) {
                            self->tmp[counter] = MVAL(lb_p, u, i);
                            ++counter;
                        }
                        qselect(self->tmp, counter, counter - lbudget - 1);
                        VVAL(lb_nbrs_v, i) = self->tmp[counter - lbudget - 1];
                    }
                }
            }
            else {
                VEC_clear(lb_nbrs_v);
                MV_min(lb_nbrs_v, lb_nbrs_v, lb_p, PGraph_piadj(self->G, v));
                if (qideg > 0)
                    MV_min(lb_nbrs_v, lb_nbrs_v, lb_p,
                           PGraph_qiadj(self->G, v));
            }
        }
        else if ((nideg > 0) && (pideg > 0)) {
            if (lbudget < pideg) {
                if (lbudget == 1) {
                    MV_max2(tmp, lb_nbrs_v, lb_p, PGraph_piadj(self->G, v));
                }
                else {
                    ITER_VEC(lb_nbrs_v, i) {
                        size_t counter = 0;
                        ITER_PINBRS(self->G, v, u) {
                            self->tmp[counter] = MVAL(lb_p, u, i);
                            ++counter;
                        }
                        qselect(self->tmp, counter, counter - lbudget - 1);
                        VVAL(lb_nbrs_v, i) = self->tmp[counter - lbudget - 1];
                    }
                }
                MV_max(lb_nbrs_v, lb_nbrs_v, lb_p, PGraph_niadj(self->G, v));
            }
            else {
                MV_max(lb_nbrs_v, NULL, lb_p, PGraph_niadj(self->G, v));
            }
        }

        // ub
        if ((nideg == 0) && (pideg == 0)) {
            VEC_clear(ub_nbrs_v);
        }
        else if ((nideg > 0) && (pideg == 0)) {
            MV_max(ub_nbrs_v, NULL, ub_p, PGraph_niadj(self->G, v));
        }
        else if ((nideg == 0) && (pideg > 0)) {
            if (lbudget < pideg) {
                MV_max(ub_nbrs_v, NULL, ub_p, PGraph_piadj(self->G, v));
            }
            else {
                VEC_clear(ub_nbrs_v);
                MV_max(ub_nbrs_v, ub_nbrs_v, ub_p, PGraph_piadj(self->G, v));
            }
        }
        else if ((nideg > 0) && (pideg > 0)) {
            MV_max(ub_nbrs_v, NULL, ub_p, PGraph_iadj(self->G, v));
        }

        if (qideg > 0)
            MV_max(ub_nbrs_v, ub_nbrs_v, ub_p, PGraph_qiadj(self->G, v));
    }

    MV_relax(self->tmp_lb_nbrs[l], self->tmp_ub_nbrs[l], CA[l], CA_ABS[l],
             lb_nbrs_v, ub_nbrs_v, self->tmp_vec1[l - 1],
             self->tmp_vec2[l - 1]);
}

static void BManager_comp_nbrs_max_loose(BManager *self, size_t l, size_t v) {
    VEC *lb_nbrs_v = self->tmp_lb_nbrs[l - 1];
    VEC *ub_nbrs_v = self->tmp_ub_nbrs[l - 1];
    MAT *lb_p      = self->lb[l - 1];
    MAT *ub_p      = self->ub[l - 1];

    size_t nideg = PGraph_nideg(self->G, v);
    size_t pideg = PGraph_pideg(self->G, v);
    size_t qideg = IS_DEL_INS ? PGraph_qideg(self->G, v) : 0;

    if (nideg > 0) {
        MV_max(lb_nbrs_v, NULL, lb_p, PGraph_niadj(self->G, v));
        MV_max(ub_nbrs_v, NULL, ub_p, PGraph_niadj(self->G, v));
        if (pideg > 0)
            MV_max(ub_nbrs_v, ub_nbrs_v, ub_p, PGraph_piadj(self->G, v));
        if (qideg > 0)
            MV_max(ub_nbrs_v, ub_nbrs_v, ub_p, PGraph_qiadj(self->G, v));
    }
    else {
        VEC_clear(lb_nbrs_v);
        VEC_clear(ub_nbrs_v);
        if (pideg > 0) {
            MV_min(lb_nbrs_v, lb_nbrs_v, lb_p, PGraph_piadj(self->G, v));
            MV_max(ub_nbrs_v, ub_nbrs_v, ub_p, PGraph_piadj(self->G, v));
        }
        if (qideg > 0) {
            MV_min(lb_nbrs_v, lb_nbrs_v, lb_p, PGraph_qiadj(self->G, v));
            MV_max(ub_nbrs_v, ub_nbrs_v, ub_p, PGraph_qiadj(self->G, v));
        }
    }

    MV_relax(self->tmp_lb_nbrs[l], self->tmp_ub_nbrs[l], CA[l], CA_ABS[l],
             lb_nbrs_v, ub_nbrs_v, self->tmp_vec1[l - 1],
             self->tmp_vec2[l - 1]);
}

static void BManager_comp_nbrs_max(BManager *self, size_t l, size_t v) {
    if (USE_TIGHT_BOUND)
        BManager_comp_nbrs_max_tight(self, l, v);
    else
        BManager_comp_nbrs_max_loose(self, l, v);
}

// ----------------------------- comp bounds mean ------------------------------
static void BManager_comp_nbrs_mean_fst(BManager *self, size_t v) {
    VEC   *lb_nbrs_v, *ub_nbrs_v;
    MAT   *feat_p;
    size_t n;
    if (USE_REORDER_COMP) {
        lb_nbrs_v = self->tmp_lb_nbrs[1];
        ub_nbrs_v = self->tmp_ub_nbrs[1];
        feat_p    = INPUT_FEAT_RXCA;
    }
    if (!USE_REORDER_COMP) {
        lb_nbrs_v = self->tmp_lb_nbrs[0];
        ub_nbrs_v = self->tmp_ub_nbrs[0];
        feat_p    = INPUT_FEAT;
    }

    // normal
    MV_acc(lb_nbrs_v, NULL, feat_p, PGraph_niadj(self->G, v));
    MV_acc(ub_nbrs_v, NULL, feat_p, PGraph_niadj(self->G, v));

    // potential
    size_t lbudget = PGraph_lbudget(self->G, v);
    size_t nideg   = PGraph_nideg(self->G, v);
    ITER_VEC(lb_nbrs_v, i) {
        size_t counter = 0;
        ITER_PINBRS(self->G, v, u) {
            self->tmp[counter] = MVAL(feat_p, u, i);
            ++counter;
        }
        qsortd(self->tmp, counter);

        // lb
        n = nideg;
        for (size_t j = 0; j < counter; ++j) {
            double val = self->tmp[j];
            if (n == 0 && val > 0.)
                break;
            if (!USE_TIGHT_BOUND || (j + lbudget >= counter))
                if (val * n > VVAL(lb_nbrs_v, i))
                    break;
            VVAL(lb_nbrs_v, i) += val;
            n++;
        }
        if (n > 0)
            VVAL(lb_nbrs_v, i) = VVAL(lb_nbrs_v, i) / n;

        // ub
        n = nideg;
        for (size_t j = 0; j < counter; ++j) {
            double val = self->tmp[counter - j - 1];
            if (n == 0 && val < 0.)
                break;
            if (!USE_TIGHT_BOUND || (j + lbudget >= counter)) {
                if (val * n < VVAL(ub_nbrs_v, i))
                    break;
            }
            VVAL(ub_nbrs_v, i) += val;
            n++;
        }
        if (n > 0)
            VVAL(ub_nbrs_v, i) = VVAL(ub_nbrs_v, i) / n;
    }

    if (!USE_REORDER_COMP)
        MV_relax(self->tmp_lb_nbrs[1], self->tmp_ub_nbrs[1], CA[1], CA_ABS[1],
                 lb_nbrs_v, ub_nbrs_v, self->tmp_vec1[0], self->tmp_vec2[0]);
}

static void BManager_comp_nbrs_mean_tail(BManager *self, size_t l, size_t v) {
    VEC   *lb_nbrs_v, *ub_nbrs_v;
    MAT   *lb_p, *ub_p;
    size_t n;
    if (USE_REORDER_COMP) {
        lb_nbrs_v = self->tmp_lb_nbrs[l];
        ub_nbrs_v = self->tmp_ub_nbrs[l];
        lb_p      = self->lb_rxcA[l];
        ub_p      = self->ub_rxcA[l];
    }
    if (!USE_REORDER_COMP) {
        lb_nbrs_v = self->tmp_lb_nbrs[l - 1];
        ub_nbrs_v = self->tmp_ub_nbrs[l - 1];
        lb_p      = self->lb[l - 1];
        ub_p      = self->ub[l - 1];
    }

    // normal
    MV_acc(lb_nbrs_v, NULL, lb_p, PGraph_niadj(self->G, v));
    MV_acc(ub_nbrs_v, NULL, ub_p, PGraph_niadj(self->G, v));

    // potential
    size_t lbudget = PGraph_lbudget(self->G, v);
    size_t nideg   = PGraph_nideg(self->G, v);
    ITER_VEC(lb_nbrs_v, i) {  // lb
        size_t counter = 0;
        ITER_PINBRS(self->G, v, u) {
            self->tmp[counter] = MVAL(lb_p, u, i);
            ++counter;
        }
        qsortd(self->tmp, counter);

        n = nideg;
        for (size_t j = 0; j < counter; ++j) {
            double val = self->tmp[j];
            if (n == 0 && val > 0.)
                break;
            if (!USE_TIGHT_BOUND || (j + lbudget >= counter)) {
                if (val * n > VVAL(lb_nbrs_v, i))
                    break;
            }
            VVAL(lb_nbrs_v, i) += val;
            ++n;
        }
        if (n > 0)
            VVAL(lb_nbrs_v, i) = VVAL(lb_nbrs_v, i) / n;
    }

    ITER_VEC(ub_nbrs_v, i) {  // ub
        size_t counter = 0;
        ITER_PINBRS(self->G, v, u) {
            self->tmp[counter] = MVAL(ub_p, u, i);
            ++counter;
        }
        qsortd(self->tmp, counter);

        n = nideg;
        for (size_t j = 0; j < counter; ++j) {
            double val = self->tmp[counter - j - 1];
            if (n == 0 && val < 0.)
                break;
            if (!USE_TIGHT_BOUND || (j + lbudget >= counter)) {
                if (val * n < VVAL(ub_nbrs_v, i))
                    break;
            }
            VVAL(ub_nbrs_v, i) += val;
            ++n;
        }
        if (n > 0)
            VVAL(ub_nbrs_v, i) = VVAL(ub_nbrs_v, i) / n;
    }

    if (!USE_REORDER_COMP)
        MV_relax(self->tmp_lb_nbrs[l], self->tmp_ub_nbrs[l], CA[l], CA_ABS[l],
                 lb_nbrs_v, ub_nbrs_v, self->tmp_vec1[l - 1],
                 self->tmp_vec2[l - 1]);
}

static void BManager_comp_nbrs_mean_delins(BManager *self, size_t l, size_t v) {
    VEC   *lb_nbrs_v, *ub_nbrs_v;
    MAT   *lb_p, *ub_p;
    size_t n;
    if (USE_REORDER_COMP) {
        lb_nbrs_v = self->tmp_lb_nbrs[l];
        ub_nbrs_v = self->tmp_ub_nbrs[l];
        lb_p      = self->lb_rxcA[l];
        ub_p      = self->ub_rxcA[l];
    }
    if (!USE_REORDER_COMP) {
        lb_nbrs_v = self->tmp_lb_nbrs[l - 1];
        ub_nbrs_v = self->tmp_ub_nbrs[l - 1];
        lb_p      = self->lb[l - 1];
        ub_p      = self->ub[l - 1];
    }

    // normal
    MV_acc(lb_nbrs_v, NULL, lb_p, PGraph_niadj(self->G, v));
    MV_acc(ub_nbrs_v, NULL, ub_p, PGraph_niadj(self->G, v));

    // potential
    size_t nideg = PGraph_nideg(self->G, v);
    ITER_VEC(lb_nbrs_v, i) {  // lb
        size_t counter = 0;
        ITER_PINBRS(self->G, v, u) {
            self->tmp[counter] = MVAL(lb_p, u, i);
            ++counter;
        }
        ITER_QINBRS(self->G, v, u) {
            self->tmp[counter] = MVAL(lb_p, u, i);
            ++counter;
        }
        qsortd(self->tmp, counter);

        n = nideg;
        for (size_t j = 0; j < counter; ++j) {
            double val = self->tmp[j];
            if (n == 0 && val > 0.)
                break;
            if (val * n > VVAL(lb_nbrs_v, i))
                break;
            VVAL(lb_nbrs_v, i) += val;
            n++;
        }
        if (n > 0)
            VVAL(lb_nbrs_v, i) = VVAL(lb_nbrs_v, i) / n;
    }

    ITER_VEC(ub_nbrs_v, i) {  // ub
        size_t counter = 0;
        ITER_PINBRS(self->G, v, u) {
            self->tmp[counter] = MVAL(ub_p, u, i);
            ++counter;
        }
        ITER_QINBRS(self->G, v, u) {
            self->tmp[counter] = MVAL(ub_p, u, i);
            ++counter;
        }
        qsortd(self->tmp, counter);

        n = nideg;
        for (size_t j = 0; j < counter; ++j) {
            double val = self->tmp[counter - j - 1];
            if (n == 0 && val < 0.)
                break;
            if (val * n < VVAL(ub_nbrs_v, i))
                break;
            VVAL(ub_nbrs_v, i) += val;
            n++;
        }
        if (n > 0)
            VVAL(ub_nbrs_v, i) = VVAL(ub_nbrs_v, i) / n;
    }

    if (!USE_REORDER_COMP)
        MV_relax(self->tmp_lb_nbrs[l], self->tmp_ub_nbrs[l], CA[l], CA_ABS[l],
                 lb_nbrs_v, ub_nbrs_v, self->tmp_vec1[l - 1],
                 self->tmp_vec2[l - 1]);
}

static void BManager_comp_nbrs_mean(BManager *self, size_t l, size_t v) {
    if (IS_DEL_ONLY) {
        if (l == 1)
            BManager_comp_nbrs_mean_fst(self, v);
        else
            BManager_comp_nbrs_mean_tail(self, l, v);
    }
    else {
        BManager_comp_nbrs_mean_delins(self, l, v);
    }
}

// -------------------------------- comp bounds --------------------------------
static void BManager_comp_self(BManager *self, size_t l, size_t v) {
    VEC *lb_self_v = self->tmp_lb_self[l];
    VEC *ub_self_v = self->tmp_ub_self[l];

    if (l == 1) {
        VEC_copy(lb_self_v, MSLICE(INPUT_FEAT_RXCC_CB, v));
        VEC_copy(ub_self_v, MSLICE(INPUT_FEAT_RXCC_CB, v));
    }
    else {
        VEC *lb_vp = MSLICE(self->lb[l - 1], v);
        VEC *ub_vp = MSLICE(self->ub[l - 1], v);
        MV_relax(lb_self_v, ub_self_v, CC[l], CC_ABS[l], lb_vp, ub_vp,
                 self->tmp_vec1[l - 1], self->tmp_vec2[l - 1]);
        VV_add(lb_self_v, lb_self_v, CB[l]);
        VV_add(ub_self_v, ub_self_v, CB[l]);
    }
}

static void BManager_comp_bound(BManager *self, size_t l, size_t v) {
    VEC *lb_v      = MSLICE(self->lb[l], v);
    VEC *ub_v      = MSLICE(self->ub[l], v);
    VEC *lb_self_v = MSLICE(self->lb_self[l], v);
    VEC *ub_self_v = MSLICE(self->ub_self[l], v);
    VEC *lb_nbrs_v = MSLICE(self->lb_nbrs[l], v);
    VEC *ub_nbrs_v = MSLICE(self->ub_nbrs[l], v);

    VV_add(lb_v, lb_self_v, lb_nbrs_v);
    VV_add(ub_v, ub_self_v, ub_nbrs_v);

    if (NOT_LAST_LAYER(l)) {
        VEC_relu(lb_v);
        VEC_relu(ub_v);
    }
    if (USE_REORDER_COMP && !IS_MAX_GNN && NOT_LAST_LAYER(l)) {
        VEC *lb_rxcA_vn = MSLICE(self->lb_rxcA[l + 1], v);
        VEC *ub_rxcA_vn = MSLICE(self->ub_rxcA[l + 1], v);
        MV_relax(lb_rxcA_vn, ub_rxcA_vn, CA[l + 1], CA_ABS[l + 1], lb_v, ub_v,
                 self->tmp_vec1[l], self->tmp_vec2[l]);
    }
}

static void BManager_comp(BManager *self) {
    ITER_LAYERS(l) {
        ITER_VTXS(v) {
            BManager_comp_self(self, l, v);
            if (IS_SUM_GNN)
                BManager_comp_nbrs_sum(self, l, v);
            if (IS_MAX_GNN)
                BManager_comp_nbrs_max(self, l, v);
            if (IS_MEAN_GNN)
                BManager_comp_nbrs_mean(self, l, v);
            MV_SWAP(self->tmp_lb_self[l], self->lb_self[l], v);
            MV_SWAP(self->tmp_ub_self[l], self->ub_self[l], v);
            MV_SWAP(self->tmp_lb_nbrs[l], self->lb_nbrs[l], v);
            MV_SWAP(self->tmp_ub_nbrs[l], self->ub_nbrs[l], v);
            BManager_comp_bound(self, l, v);
        }
    }
    if (IS_GRAPH_CLASS) {
        MV_acc(self->ub_pool, NULL, self->ub[N_LAYERS], NULL);
        VV_add(self->ub_pool, self->ub_pool, CBL);
    }
}

// ------------------------------- update bounds -------------------------------
static void BManager_update_single(BManager *self, size_t l, size_t v) {
    bool is_dirty_self = BArray_test(self->dirty_self, v);
    bool is_dirty_nbrs = BArray_test(self->dirty_nbrs, v);
    if (!is_dirty_self && !is_dirty_nbrs)
        return;

    // comp
    if (is_dirty_self)
        BManager_comp_self(self, l, v);
    if (IS_SUM_GNN)
        BManager_comp_nbrs_sum(self, l, v);
    if (IS_MAX_GNN)
        BManager_comp_nbrs_max(self, l, v);
    if (IS_MEAN_GNN)
        BManager_comp_nbrs_mean(self, l, v);

    // push
    if (IS_GRAPH_CLASS && IS_LAST_LAYER(l))
        VV_sub(self->ub_pool, self->ub_pool, MSLICE(self->ub[l], v));
    BManager_push_snap(self, l, v, is_dirty_self, is_dirty_nbrs);

    // swap and comp
    if (is_dirty_self) {
        MV_SWAP(self->tmp_lb_self[l], self->lb_self[l], v);
        MV_SWAP(self->tmp_ub_self[l], self->ub_self[l], v);
    }
    if (is_dirty_nbrs) {
        MV_SWAP(self->tmp_lb_nbrs[l], self->lb_nbrs[l], v);
        MV_SWAP(self->tmp_ub_nbrs[l], self->ub_nbrs[l], v);
    }
    BManager_comp_bound(self, l, v);

    // update pool
    if (IS_GRAPH_CLASS && IS_LAST_LAYER(l))
        VV_add(self->ub_pool, self->ub_pool, MSLICE(self->ub[l], v));

    // propagate
    if (NOT_LAST_LAYER(l)) {
        BArray_set(self->dirty_self_next, v);
        BArray_union(self->dirty_nbrs_next, PGraph_oadj(self->G, v));
    }
}

static void BManager_update(BManager *self) {
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
        Op_type   op_type   = PGraph_op_type(self->G);
        // nodes with edges changed
        BArray_union(self->dirty_nbrs, PGraph_updated_edges(self->G));
        // nodes with local budgets changed
        if ((edge_type == PEDGE) && (op_type == CUT))
            BArray_union(self->dirty_nbrs, PGraph_updated_budgets(self->G));
        if ((edge_type == QEDGE) && (op_type == CON))
            BArray_union(self->dirty_nbrs, PGraph_updated_budgets(self->G));

        // update bounds
        if (IS_NODE_CLASS) {
            ITER_KNBRS(self->G, N_LAYERS - l, v) {
                BManager_update_single(self, l, v);
            }
        }
        if (IS_GRAPH_CLASS) {
            ITER_VTXS(v) {
                BManager_update_single(self, l, v);
            }
        }
    }
}

// ---------------------------------- status -----------------------------------
void BManager_init(BManager *self) {
    // allocate
    self->lb          = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->ub          = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->lb_self     = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->ub_self     = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->lb_nbrs     = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->ub_nbrs     = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->lb_rxcA     = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->ub_rxcA     = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    self->tmp_lb_self = XMALLOC(N_LAYERS_EXT * sizeof(VEC *));
    self->tmp_ub_self = XMALLOC(N_LAYERS_EXT * sizeof(VEC *));
    self->tmp_lb_nbrs = XMALLOC(N_LAYERS_EXT * sizeof(VEC *));
    self->tmp_ub_nbrs = XMALLOC(N_LAYERS_EXT * sizeof(VEC *));
    self->tmp_vec1    = XMALLOC(N_LAYERS_EXT * sizeof(VEC *));
    self->tmp_vec2    = XMALLOC(N_LAYERS_EXT * sizeof(VEC *));

    ITER_LAYERS(l) {
        self->lb[l]      = MAT_alloc(N_VERTICES, DIM[l]);
        self->ub[l]      = MAT_alloc(N_VERTICES, DIM[l]);
        self->lb_self[l] = MAT_alloc(N_VERTICES, DIM[l]);
        self->ub_self[l] = MAT_alloc(N_VERTICES, DIM[l]);
        self->lb_nbrs[l] = MAT_alloc(N_VERTICES, DIM[l]);
        self->ub_nbrs[l] = MAT_alloc(N_VERTICES, DIM[l]);
        if (l == 1)
            continue;
        self->lb_rxcA[l] = MAT_alloc(N_VERTICES, DIM[l]);
        self->ub_rxcA[l] = MAT_alloc(N_VERTICES, DIM[l]);
    }
    ITER_LAYERS_EXT(l) {
        self->tmp_lb_self[l] = VEC_alloc(DIM[l]);
        self->tmp_ub_self[l] = VEC_alloc(DIM[l]);
        self->tmp_lb_nbrs[l] = VEC_alloc(DIM[l]);
        self->tmp_ub_nbrs[l] = VEC_alloc(DIM[l]);
        self->tmp_vec1[l]    = VEC_alloc(DIM[l]);
        self->tmp_vec2[l]    = VEC_alloc(DIM[l]);
    }
    self->ub_pool = VEC_alloc(DIM_LAST);

    self->tmp = XMALLOC(N_VERTICES * sizeof(double));

    self->dirty_self      = BArray_alloc(N_VERTICES);
    self->dirty_self_next = BArray_alloc(N_VERTICES);
    self->dirty_nbrs      = BArray_alloc(N_VERTICES);
    self->dirty_nbrs_next = BArray_alloc(N_VERTICES);

    self->stack =
        Stack_alloc(sizeof(BManagerSnap), N_LAYERS, BManager_snap_free);
    self->stack_pool =
        Stack_alloc(sizeof(BManagerPoolSnap), 1, BManager_poolsnap_free);

    self->clock = 0;

    // init
    self->init       = true;
    self->lb[0]      = INPUT_FEAT;
    self->ub[0]      = INPUT_FEAT;
    self->lb_rxcA[1] = INPUT_FEAT_RXCA;
    self->ub_rxcA[1] = INPUT_FEAT_RXCA;
    BManager_comp(self);
}

void BManager_push(BManager *self) {
    Stack_push_marker(self->stack);

    if (IS_GRAPH_CLASS) {
        BManagerPoolSnap *snap =
            (BManagerPoolSnap *)Stack_push(self->stack_pool, 1);
        if (Stack_fresh(self->stack_pool))
            snap->ub_pool = VEC_alloc(DIM_LAST);
        VEC_copy(snap->ub_pool, self->ub_pool);
    }
}

void BManager_pop(BManager *self) {
    BManagerSnap *snap;
    while ((snap = (BManagerSnap *)Stack_pop(self->stack)) != NULL) {
        size_t l = snap->l;
        size_t v = snap->v;
        MV_SWAP(snap->lb_v, self->lb[l], v);
        MV_SWAP(snap->ub_v, self->ub[l], v);
        if (snap->push_self) {
            MV_SWAP(snap->lb_self_v, self->lb_self[l], v);
            MV_SWAP(snap->ub_self_v, self->ub_self[l], v);
        }
        if (snap->push_nbrs) {
            MV_SWAP(snap->lb_nbrs_v, self->lb_nbrs[l], v);
            MV_SWAP(snap->ub_nbrs_v, self->ub_nbrs[l], v);
        }
        if (NOT_LAST_LAYER(l)) {
            MV_SWAP(snap->lb_rxcA_v, self->lb_rxcA[l + 1], v);
            MV_SWAP(snap->ub_rxcA_v, self->ub_rxcA[l + 1], v);
        }
    }

    if (IS_GRAPH_CLASS) {
        BManagerPoolSnap *snap =
            (BManagerPoolSnap *)Stack_pop(self->stack_pool);
        VEC *tmp      = snap->ub_pool;
        snap->ub_pool = self->ub_pool;
        self->ub_pool = tmp;
    }
}

void BManager_flush(BManager *self) {
    BManager_comp(self);
}

// ------------------------------------ sat ------------------------------------
bool BManager_unsat(BManager *self) {
    PROFILING_START;

    if (self->init)
        self->init = false;
    else {
        if (USE_INC_COMP)
            BManager_update(self);
        else
            BManager_comp(self);
    }

    bool ret = false;
    if (IS_NODE_CLASS)
        ret = !VEC_any_pos(MSLICE(self->ub[N_LAYERS], 0));
    if (IS_GRAPH_CLASS)
        ret = !VEC_any_pos(self->ub_pool);

    PROFILING_END;

    return ret;
}
