#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "Graph.h"
#include "Matrix.h"
#include "utils.h"

// ---------------------------------- parser -----------------------------------
static class_t parse_class_t(const char *str) {
    if (strcmp(str, "NODE") == 0)
        return CLASS_NODE;
    else if (strcmp(str, "GRAPH") == 0)
        return CLASS_GRAPH;
    else
        CHECK(true, "unknown classification %s!", str);
}

static aggr_t parse_aggr_t(const char *str) {
    if (strcmp(str, "SUM") == 0)
        return AGGR_SUM;
    else if (strcmp(str, "MAX") == 0)
        return AGGR_MAX;
    else if (strcmp(str, "MEAN") == 0)
        return AGGR_MEAN;
    else
        CHECK(true, "unknown aggregation %s!", str);
}

static graph_t parse_graph_t(const char *str) {
    if (strcmp(str, "DIRECTED") == 0)
        return GRAPH_DIRECTED;
    else if (strcmp(str, "UNDIRECTED") == 0)
        return GRAPH_UNDIRECTED;
    else
        CHECK(true, "unknown direction %s!", str);
}

static pert_t parse_pert_t(const char *str) {
    if (strcmp(str, "DEL_ONLY") == 0)
        return PERT_DEL_ONLY;
    else if (strcmp(str, "DEL_INS") == 0)
        return PERT_DEL_INS;
    else
        CHECK(true, "unknown perturbation %s!", str);
}

// -------------------------------- evaluation ---------------------------------
static void Common_eval_layer_sum(MAT feat, MAT feat_prev, size_t l) {
    MAT feat_prev_rxcA = MAT_alloc(N_VERTICES, DIM[l]);
    MM_dot_trans(feat_prev_rxcA, feat_prev, CA[l], NULL, NULL, 0);

    ITER_VTXS(v) {
        VEC       feat_v = feat[v];
        uint16_t *list   = HGraph_glist(INPUT_GRAPH, v);
        size_t    deg    = HGraph_gdeg(INPUT_GRAPH, v);

        MV_acc(feat_v, feat_v, feat_prev_rxcA, list, deg);
    }

    MAT_free(feat_prev_rxcA);
}

static void Common_eval_layer_max(MAT feat, MAT feat_prev, size_t l) {
    VEC tmp      = VEC_alloc(DIM[l - 1]);
    VEC tmp_rxcA = VEC_alloc(DIM[l]);

    ITER_VTXS(v) {
        VEC       feat_v = feat[v];
        uint16_t *list   = HGraph_glist(INPUT_GRAPH, v);
        size_t    deg    = HGraph_gdeg(INPUT_GRAPH, v);

        if (deg > 0) {
            MV_max(tmp, NULL, feat_prev, list, deg);
            MV_dot(tmp_rxcA, CA[l], tmp, NULL);
            VV_add(feat_v, feat_v, tmp_rxcA);
        }
    }

    VEC_free(tmp);
    VEC_free(tmp_rxcA);
}

static void Common_eval_layer_mean(MAT feat, MAT feat_prev, size_t l) {
    MAT feat_prev_rxcA = MAT_alloc(N_VERTICES, DIM[l]);
    VEC tmp            = VEC_alloc(DIM[l]);
    MM_dot_trans(feat_prev_rxcA, feat_prev, CA[l], NULL, NULL, 0);

    ITER_VTXS(v) {
        VEC       feat_v = feat[v];
        uint16_t *list   = HGraph_glist(INPUT_GRAPH, v);
        size_t    deg    = HGraph_gdeg(INPUT_GRAPH, v);

        if (deg > 0) {
            MV_acc(tmp, NULL, feat_prev_rxcA, list, deg);
            VEC_div(tmp, deg);
            VV_add(feat_v, feat_v, tmp);
        }
    }

    MAT_free(feat_prev_rxcA);
    VEC_free(tmp);
}

static MAT Common_eval_layer(MAT feat_prev, size_t l) {
    MAT feat = MAT_alloc(N_VERTICES, DIM[l]);
    MM_dot_trans(feat, feat_prev, CC[l], CB[l], NULL, 0);

    if (IS_SUM_GNN)
        Common_eval_layer_sum(feat, feat_prev, l);
    if (IS_MAX_GNN)
        Common_eval_layer_max(feat, feat_prev, l);
    if (IS_MEAN_GNN)
        Common_eval_layer_mean(feat, feat_prev, l);

    if (NOT_LAST_LAYER(l)) {
        ITER_VTXS(v) {
            VEC_relu(feat[v]);
        }
    }

    return feat;
}

size_t Common_eval() {
    MAT feat_prev = INPUT_FEAT;
    MAT feat;

    ITER_LAYERS(l) {
        feat = Common_eval_layer(feat_prev, l);
        if (l > 1)
            MAT_free(feat_prev);
        feat_prev = feat;
    }

    size_t ret = 0;
    if (IS_NODE_CLASS) {
        ret = VEC_argmax(feat_prev[0]);
    }
    if (IS_GRAPH_CLASS) {
        VEC pool_prev = VEC_alloc(DIM_LAST);
        VEC pool      = VEC_alloc(DIM[N_LAYERS_EXT]);

        MV_acc(pool_prev, NULL, feat_prev, NULL, 0);
        MV_dot(pool, CAP, pool_prev, CBP);
        ret = VEC_argmax(pool);

        VEC_free(pool_prev);
        VEC_free(pool);
    }

    MAT_free(feat_prev);

    return ret;
}

// --------------------------------- lifecycle ---------------------------------
Common common;

void Common_init(const char *gnn_path, const char *graph_path,
                 const char *feat_path, const char *pert, size_t variant,
                 size_t ori, size_t shift) {
    FILE *fp;
    char  buf[100];

    // read GNN
    fp = XFOPEN(gnn_path, "r");

    fscanf(fp, "%s\n", buf);
    common.class = parse_class_t(buf);
    fscanf(fp, "%s\n", buf);
    common.aggr = parse_aggr_t(buf);
    fscanf(fp, "%lu", &N_LAYERS);

    DIM = XMALLOC((N_LAYERS_EXT + 1) * sizeof(size_t));

    ITER_LAYERS_EXT(l) {
        fscanf(fp, "%lu", DIM + l);
    }
    if (IS_GRAPH_CLASS) {
        fscanf(fp, "%lu", DIM + N_LAYERS_EXT);
    }

    CC = XMALLOC(N_LAYERS_EXT * sizeof(MAT));
    CA = XMALLOC(N_LAYERS_EXT * sizeof(MAT));
    CB = XMALLOC(N_LAYERS_EXT * sizeof(VEC));

    ITER_LAYERS(l) {
        CC[l] = MAT_alloc_fp(DIM[l], DIM[l - 1], fp);
        CA[l] = MAT_alloc_fp(DIM[l], DIM[l - 1], fp);
        CB[l] = VEC_alloc_fp(DIM[l], fp);
    }
    if (IS_GRAPH_CLASS) {
        CAP = MAT_alloc_fp(DIM[N_LAYERS_EXT], DIM_LAST, fp);
        CBP = VEC_alloc_fp(DIM[N_LAYERS_EXT], fp);
    }

    fclose(fp);

    // read graph
    fp = XFOPEN(graph_path, "r");

    fscanf(fp, "%lu\n", &N_VERTICES);
    fscanf(fp, "%s\n", buf);
    common.directed    = parse_graph_t(buf);
    common.input_graph = HGraph_alloc_fp(fp);
    common.n_edges     = HGraph_count_edges(INPUT_GRAPH);

    fclose(fp);

    // read feature
    INPUT_FEAT         = MAT_alloc_file(N_VERTICES, DIM[0], feat_path);
    INPUT_FEAT_RXCA    = MAT_alloc(N_VERTICES, DIM[1]);
    INPUT_FEAT_RXCC_CB = MAT_alloc(N_VERTICES, DIM[1]);

    // set perturbation
    common.pert = parse_pert_t(pert);

    // set options
    common.inc_comp       = !(variant & 1);
    common.reorder_comp   = !(variant & 2);
    common.tight_bound    = !(variant & 4);
    common.heuristic_pick = !(variant & 8);

    // comp original and target classes
    ORI = ori >= DIM_LAST ? Common_eval() : ori;
    TGT = (ORI + shift) % DIM_LAST;

    // comp effective coefficients for graph classification
    if (IS_GRAPH_CLASS) {
        MAT tmp_cC = MAT_alloc(DIM[N_LAYERS_EXT], DIM[N_LAYERS - 1]);
        MAT tmp_cA = MAT_alloc(DIM[N_LAYERS_EXT], DIM[N_LAYERS - 1]);
        VEC tmp_cb = VEC_alloc(DIM[N_LAYERS_EXT]);
        MM_dot(tmp_cC, CAP, CC[N_LAYERS]);
        MM_dot(tmp_cA, CAP, CA[N_LAYERS]);
        MV_dot(tmp_cb, CAP, CB[N_LAYERS], NULL);

        MAT_free(CC[N_LAYERS]);
        MAT_free(CA[N_LAYERS]);
        VEC_free(CB[N_LAYERS]);
        CC[N_LAYERS] = tmp_cC;
        CA[N_LAYERS] = tmp_cA;
        CB[N_LAYERS] = tmp_cb;

        DIM_LAST = DIM[N_LAYERS_EXT];
    }

    // comp effective coefficients for robustness
    MAT_shift(CC[N_LAYERS], ORI, shift);
    MAT_shift(CA[N_LAYERS], ORI, shift);
    VEC_shift(CB[N_LAYERS], ORI, shift);
    if (IS_GRAPH_CLASS)
        VEC_shift(CBP, ORI, shift);

    if (shift == 0)
        DIM_LAST -= 1;
    else
        DIM_LAST = 1;

    // comp maximum dim
    DIM_MAX = 0;
    ITER_LAYERS_EXT(l) {
        DIM_MAX = MAX(DIM_MAX, DIM[l]);
    }

    // comp abs
    ITER_LAYERS(l) {
        MAT_comp_abs(CC[l]);
        MAT_comp_abs(CA[l]);
    }

    // comp relaxation of input feat
    MM_dot_trans(INPUT_FEAT_RXCA, INPUT_FEAT, CA[1], NULL, NULL, 0);
    MM_dot_trans(INPUT_FEAT_RXCC_CB, INPUT_FEAT, CC[1], CB[1], NULL, 0);
}

void Common_cleanup() {
    // GNN
    free(DIM);

    ITER_LAYERS(l) {
        MAT_free(CC[l]);
        MAT_free(CA[l]);
        VEC_free(CB[l]);
    }
    if (IS_GRAPH_CLASS) {
        MAT_free(CAP);
        VEC_free(CBP);
    }

    free(CC);
    free(CA);
    free(CB);

    // graph
    HGraph_free(INPUT_GRAPH);

    // feature
    MAT_free(INPUT_FEAT);
    MAT_free(INPUT_FEAT_RXCA);
    MAT_free(INPUT_FEAT_RXCC_CB);
}

void Common_dump() {
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
void Common_write_perturbation(FILE *fp) {
    if (IS_DEL_ONLY)
        fprintf(fp, "DELETION ONLY\n");
    if (IS_DEL_INS)
        fprintf(fp, "DELETION AND INSERTION\n");
}

void Common_write_variant(FILE *fp) {
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
