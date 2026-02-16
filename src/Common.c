#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "Graph.h"
#include "Matrix.h"
#include "utils.h"

// ---------------------------------- parser -----------------------------------
static Class_type parse_class_type(const char *str) {
    if (strcmp(str, "NODE") == 0)
        return CLASS_NODE;
    else if (strcmp(str, "GRAPH") == 0)
        return CLASS_GRAPH;
    else
        CHECK(true, "unknown classification %s!", str);
}

static Aggr_type parse_aggr_type(const char *str) {
    if (strcmp(str, "SUM") == 0)
        return AGGR_SUM;
    else if (strcmp(str, "MAX") == 0)
        return AGGR_MAX;
    else if (strcmp(str, "MEAN") == 0)
        return AGGR_MEAN;
    else
        CHECK(true, "unknown aggregation %s!", str);
}

static Graph_type parse_graph_type(const char *str) {
    if (strcmp(str, "DIRECTED") == 0)
        return GRAPH_DIRECTED;
    else if (strcmp(str, "UNDIRECTED") == 0)
        return GRAPH_UNDIRECTED;
    else
        CHECK(true, "unknown direction %s!", str);
}

static Pert_type parse_pert_type(const char *str) {
    if (strcmp(str, "DEL_ONLY") == 0)
        return PERT_DEL_ONLY;
    else if (strcmp(str, "DEL_INS") == 0)
        return PERT_DEL_INS;
    else
        CHECK(true, "unknown perturbation %s!", str);
}

// --------------------------------- lifecycle ---------------------------------
Common *Common_alloc(const char *gnn_path, const char *graph_path,
                     const char *feat_path, const char *pert, size_t variant) {
    Common *self = XMALLOC(sizeof(Common));
    self->common = self;

    FILE *fp;
    char  buf[100];

    // GNN
    fp = XFOPEN(gnn_path, "r");

    fscanf(fp, "%s\n", buf);
    self->class = parse_class_type(buf);
    fscanf(fp, "%s\n", buf);
    self->aggr = parse_aggr_type(buf);
    fscanf(fp, "%lu", &N_LAYERS);

    DIM = XMALLOC(N_LAYERS_EXT * sizeof(size_t));

    ITER_LAYERS_EXT(l) {
        fscanf(fp, "%lu", DIM + l);
    }
    if (IS_GRAPH_CLASS) {
        fscanf(fp, "%lu", &DIML);
    }

    CC = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    CA = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    CB = XMALLOC(N_LAYERS_EXT * sizeof(VEC *));

    ITER_LAYERS(l) {
        CC[l] = MAT_alloc_fp(DIM[l], DIM[l - 1], fp);
        CA[l] = MAT_alloc_fp(DIM[l], DIM[l - 1], fp);
        CB[l] = VEC_alloc_fp(DIM[l], fp);
    }
    if (IS_GRAPH_CLASS) {
        CAL = MAT_alloc_fp(DIML, DIM_LAST, fp);
        CBL = VEC_alloc_fp(DIML, fp);
    }

    fclose(fp);

    // Graph
    fp = XFOPEN(graph_path, "r");

    fscanf(fp, "%lu\n", &N_VERTICES);
    fscanf(fp, "%s\n", buf);
    self->directed = parse_graph_type(buf);
    self->G        = Graph_alloc_fp(self->common, fp);
    self->n_edges  = Graph_count_edges(self->G);

    fclose(fp);

    self->input_feat         = MAT_alloc_file(N_VERTICES, DIM[0], feat_path);
    self->input_feat_rxcA    = MAT_alloc(N_VERTICES, DIM[1]);
    self->input_feat_rxcC_cb = MAT_alloc(N_VERTICES, DIM[1]);

    // Perturbation
    self->pert = parse_pert_type(pert);

    // Options
    self->inc_comp       = !(variant & 1);
    self->reorder_comp   = !(variant & 2);
    self->tight_bound    = !(variant & 4);
    self->heuristic_pick = !(variant & 8);

    return self;
}

void Common_free(Common *self) {
    // GNN
    free(DIM);

    ITER_LAYERS(l) {
        MAT_free(CC[l]);
        MAT_free(CA[l]);
        VEC_free(CB[l]);
        MAT_free(CC_ABS[l]);
        MAT_free(CA_ABS[l]);
    }
    if (IS_GRAPH_CLASS) {
        MAT_free(CAL);
        VEC_free(CBL);
    }

    free(CC);
    free(CA);
    free(CB);
    free(CC_ABS);
    free(CA_ABS);

    // Graph
    Graph_free(INPUT_G);

    MAT_free(self->input_feat);
    MAT_free(self->input_feat_rxcA);
    MAT_free(self->input_feat_rxcC_cb);

    free(self);
}

void Common_dump(Common *self) {
    printf("#layers: %ld\n", N_LAYERS);
    printf("dim: ");
    ITER_LAYERS_EXT(l) {
        printf("%ld, ", DIM[l]);
    }
    printf("\n");

    ITER_LAYERS(l) {
        printf("cC[%ld]: \n", l);
        MAT_dump(CC[l]);

        printf("cA[%ld]: \n", l);
        MAT_dump(CA[l]);

        printf("cb[%ld]: \n", l);
        VEC_dump(CB[l]);
    }
    printf("\n");
}

// ----------------------------------- info ------------------------------------
void Common_dump_perturbation(Common *self, FILE *fp) {
    if (IS_DEL_ONLY)
        fprintf(fp, "DELETION ONLY\n");
    if (IS_DEL_INS)
        fprintf(fp, "DELETION AND INSERTION\n");
}

void Common_dump_variant(Common *self, FILE *fp) {
    if (!USE_INC_COMP)
        fprintf(fp, "NO_INC_COMP, ");
    if (!USE_REORDER_COMP)
        fprintf(fp, "NO_REORDER_COMP, ");
    if (!USE_TIGHT_BOUND)
        fprintf(fp, "NO_TIGHT_BOUND, ");
    if (!USE_HEURISTIC_PICK)
        fprintf(fp, "NO_HEURISTIC_PICK, ");
    fprintf(fp, "\n");
}

// -------------------------------- evaluation ---------------------------------
static void Common_eval_layer_sum(Common *self, MAT *feat, MAT *feat_prev,
                                  size_t l) {
    MAT *feat_prev_rxcA = MAT_alloc(N_VERTICES, DIM[l]);
    MM_dot_trans(feat_prev_rxcA, feat_prev, CA[l]);

    ITER_VTXS(v) {
        VEC *feat_v = MSLICE(feat, v);
        MV_acc(feat_v, feat_v, feat_prev_rxcA, Graph_iadj(self->G, v));
    }

    MAT_free(feat_prev_rxcA);
}

static void Common_eval_layer_max(Common *self, MAT *feat, MAT *feat_prev,
                                  size_t l) {
    VEC *tmp      = VEC_alloc(DIM[l - 1]);
    VEC *tmp_rxcA = VEC_alloc(DIM[l]);

    ITER_VTXS(v) {
        VEC *feat_v = MSLICE(feat, v);
        if (Graph_ideg(self->G, v) > 0) {
            MV_max(tmp, NULL, feat_prev, Graph_iadj(self->G, v));
            MV_dot(tmp_rxcA, CA[l], tmp);
            VV_add(feat_v, feat_v, tmp_rxcA);
        }
    }

    VEC_free(tmp);
    VEC_free(tmp_rxcA);
}

static void Common_eval_layer_mean(Common *self, MAT *feat, MAT *feat_prev,
                                   size_t l) {
    MAT *feat_prev_rxcA = MAT_alloc(N_VERTICES, DIM[l]);
    VEC *tmp            = VEC_alloc(DIM[l]);
    MM_dot_trans(feat_prev_rxcA, feat_prev, CA[l]);

    ITER_VTXS(v) {
        VEC *feat_v = MSLICE(feat, v);
        if (Graph_ideg(self->G, v) > 0) {
            MV_acc(tmp, NULL, feat_prev_rxcA, Graph_iadj(self->G, v));
            VEC_div(tmp, Graph_ideg(self->G, v));
            VV_add(feat_v, feat_v, tmp);
        }
    }

    MAT_free(feat_prev_rxcA);
    VEC_free(tmp);
}

static MAT *Common_eval_layer(Common *self, MAT *feat_prev, size_t l) {
    MAT *feat = MAT_alloc(N_VERTICES, DIM[l]);
    MM_dot_trans(feat, feat_prev, CC[l]);

    if (IS_SUM_GNN)
        Common_eval_layer_sum(self, feat, feat_prev, l);
    if (IS_MAX_GNN)
        Common_eval_layer_max(self, feat, feat_prev, l);
    if (IS_MEAN_GNN)
        Common_eval_layer_mean(self, feat, feat_prev, l);

    ITER_VTXS(v) {
        VEC *feat_v = MSLICE(feat, v);
        VV_add(feat_v, feat_v, CB[l]);
        if (NOT_LAST_LAYER(l))
            VEC_relu(feat_v);
    }

    return feat;
}

size_t Common_eval(Common *self) {
    MAT *feat_prev = self->input_feat;
    MAT *feat;

    ITER_LAYERS(l) {
        feat = Common_eval_layer(self, feat_prev, l);
        if (l > 1)
            MAT_free(feat_prev);
        feat_prev = feat;
    }

    size_t ret = 0;
    if (IS_NODE_CLASS) {
        ret = VEC_argmax(MSLICE(feat_prev, 0));
    }
    if (IS_GRAPH_CLASS) {
        VEC *pool     = VEC_alloc(DIM_LAST);
        VEC *pool_lin = VEC_alloc(DIML);

        MV_acc(pool, NULL, feat_prev, NULL);
        MV_dot(pool_lin, CAL, pool);
        VV_add(pool_lin, pool_lin, CBL);
        ret = VEC_argmax(pool_lin);

        VEC_free(pool);
        VEC_free(pool_lin);
    }

    MAT_free(feat_prev);

    return ret;
}

// ---------------------------------- status -----------------------------------
void Common_init(Common *self, size_t ori, size_t shift) {
    // comp original and target classes
    if (ori >= DIM_LAST)
        ori = Common_eval(self);
    self->ori = ori;
    self->tgt = (ori + shift) % DIM_LAST;

    // comp effective coefficients for graph classification
    if (IS_GRAPH_CLASS) {
        MAT *tmp_cC = MAT_alloc(DIML, DIM[N_LAYERS - 1]);
        MAT *tmp_cA = MAT_alloc(DIML, DIM[N_LAYERS - 1]);
        VEC *tmp_cb = VEC_alloc(DIML);
        MM_dot(tmp_cC, CAL, CC[N_LAYERS]);
        MM_dot(tmp_cA, CAL, CA[N_LAYERS]);
        MV_dot(tmp_cb, CAL, CB[N_LAYERS]);

        MAT_free(CC[N_LAYERS]);
        MAT_free(CA[N_LAYERS]);
        VEC_free(CB[N_LAYERS]);
        CC[N_LAYERS] = tmp_cC;
        CA[N_LAYERS] = tmp_cA;
        CB[N_LAYERS] = tmp_cb;

        DIM_LAST = DIML;
    }

    // comp effective coefficients for robustness
    MAT_shift(CC[N_LAYERS], ori, shift);
    MAT_shift(CA[N_LAYERS], ori, shift);
    VEC_shift(CB[N_LAYERS], ori, shift);
    if (IS_GRAPH_CLASS)
        VEC_shift(CBL, ori, shift);

    if (shift == 0)
        DIM_LAST -= 1;
    else
        DIM_LAST = 1;

    // comp abs of coefficients
    CC_ABS = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));
    CA_ABS = XMALLOC(N_LAYERS_EXT * sizeof(MAT *));

    ITER_LAYERS(l) {
        CC_ABS[l] = MAT_alloc(DIM[l], DIM[l - 1]);
        CA_ABS[l] = MAT_alloc(DIM[l], DIM[l - 1]);

        ITER_MAT1(CC[l], i) {
            ITER_MAT2(CC[l], j) {
                MVAL(CC_ABS[l], i, j) = ABS(MVAL(CC[l], i, j));
                MVAL(CA_ABS[l], i, j) = ABS(MVAL(CA[l], i, j));
            }
        }
    }

    // comp relaxation of input feat
    MM_dot_trans(self->input_feat_rxcA, self->input_feat, CA[1]);
    MM_dot_trans(self->input_feat_rxcC_cb, self->input_feat, CC[1]);
    ITER_VTXS(v) {
        VEC *tmp = MSLICE(self->input_feat_rxcC_cb, v);
        VV_add(tmp, tmp, CB[1]);
    }
}
