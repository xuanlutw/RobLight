#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "BudManager.h"
#include "Common.h"
#include "GManager.h"
#include "Graph.h"
#include "Stack.h"
#include "utils.h"

// ----------------------------------- type ------------------------------------
typedef struct {
    Node node;

    edge_t edge;
    size_t v;
    size_t u;
    op_t   op;
} GManagerEdgeOp;

// --------------------------------- liftcycle ---------------------------------
GManager *GManager_alloc(bool comp_radius, size_t gbudget, size_t lbudget) {
    GManager *self = XMALLOC(sizeof(GManager));

    HGraph_init_copy(&self->inc_graph, INPUT_GRAPH);
    if (IS_DEL_INS)
        HGraph_fill(&self->inc_graph);

    if (IS_DIRECTED)
        HGraph_init_copy_trans(&self->out_graph, &self->inc_graph);
    else {
        assert(IS_UNDIRECTED);
        HGraph_alias(&self->out_graph, &self->inc_graph);
    }

    self->flag_b  = XMALLOC(N_VERTICES * sizeof(bool));
    self->queue_b = XMALLOC(N_VERTICES * sizeof(uint16_t));
    self->lens_b  = XMALLOC(N_VERTICES * sizeof(uint16_t));

    self->flag_f  = XMALLOC(N_VERTICES * sizeof(bool));
    self->queue_f = XMALLOC(N_VERTICES * sizeof(uint16_t));
    self->lens_f  = XMALLOC(N_VERTICES * sizeof(uint16_t));

    BudManager_init(&self->budm, comp_radius, gbudget, lbudget);
    ITER_VTXS(v) {
        size_t deg =
            HGraph_pdeg(&self->inc_graph, v) + HGraph_qdeg(&self->inc_graph, v);
        BudManager_init_v(&self->budm, v, deg);
    }

    Stack_init(&self->stack, sizeof(GManagerEdgeOp));

    self->clock = 0;

    return self;
}

void GManager_free(GManager *self) {
    HGraph_cleanup(&self->inc_graph);
    if (IS_DIRECTED)
        HGraph_cleanup(&self->out_graph);

    free(self->flag_b);
    free(self->queue_b);
    free(self->lens_b);

    free(self->flag_f);
    free(self->queue_f);
    free(self->lens_f);

    BudManager_cleanup(&self->budm);

    Stack_cleanup(&self->stack, NULL);

    free(self);
}

void GManager_dump(GManager *self) {
    printf("  ");
    ITER_VTXS(u) {
        printf("%2ld", u);
    }
    printf("\n");

    ITER_VTXS(v) {
        printf("%2ld", v);
        ITER_VTXS(u) {
            switch (HGraph_edge(&self->inc_graph, v, u)) {
            case PEDGE:
                printf(" p");
                break;
            case NEDGE:
                printf(" n");
                break;
            case QEDGE:
                printf(" q");
                break;
            default:
                printf("  ");
                break;
            }
        }
        printf(" %2ld", GManager_pideg(self, v));
        printf(" %2ld", GManager_nideg(self, v));
        printf(" %2ld", GManager_qideg(self, v));
        printf("\n");
    }

    printf("  ");
    ITER_VTXS(u) {
        printf("%2ld", HGraph_pdeg(&self->out_graph, u));
    }
    printf("\n");

    printf("  ");
    ITER_VTXS(u) {
        printf("%2ld", HGraph_ndeg(&self->out_graph, u));
    }
    printf("\n");

    printf("  ");
    ITER_VTXS(u) {
        printf("%2ld", HGraph_qdeg(&self->out_graph, u));
    }
    printf("\n");

    printf("%2ld", GManager_gbudget(self));
    ITER_VTXS(u) {
        printf("%2ld", GManager_lbudget(self, u));
    }
    printf("\n");
}

// ------------------------------ edge operations ------------------------------
static inline void GManager_nedge_to_pedge(GManager *self, size_t v, size_t u) {
    HGraph_nedge_to_pedge(&self->inc_graph, v, u);
    if (IS_DIRECTED || (v != u))
        HGraph_nedge_to_pedge(&self->out_graph, u, v);
}

static inline void GManager_pedge_to_nedge(GManager *self, size_t v, size_t u) {
    HGraph_pedge_to_nedge(&self->inc_graph, v, u);
    if (IS_DIRECTED || (v != u))
        HGraph_pedge_to_nedge(&self->out_graph, u, v);
}

static inline void GManager_pedge_to_xedge(GManager *self, size_t v, size_t u) {
    HGraph_pedge_to_xedge(&self->inc_graph, v, u);
    if (IS_DIRECTED || (v != u))
        HGraph_pedge_to_xedge(&self->out_graph, u, v);
}

static inline void GManager_xedge_to_pedge(GManager *self, size_t v, size_t u) {
    HGraph_xedge_to_pedge(&self->inc_graph, v, u);
    if (IS_DIRECTED || (v != u))
        HGraph_xedge_to_pedge(&self->out_graph, u, v);
}

static inline void GManager_nedge_to_qedge(GManager *self, size_t v, size_t u) {
    HGraph_nedge_to_qedge(&self->inc_graph, v, u);
    if (IS_DIRECTED || (v != u))
        HGraph_nedge_to_qedge(&self->out_graph, u, v);
}

static inline void GManager_qedge_to_nedge(GManager *self, size_t v, size_t u) {
    HGraph_qedge_to_nedge(&self->inc_graph, v, u);
    if (IS_DIRECTED || (v != u))
        HGraph_qedge_to_nedge(&self->out_graph, u, v);
}

static inline void GManager_qedge_to_xedge(GManager *self, size_t v, size_t u) {
    HGraph_qedge_to_xedge(&self->inc_graph, v, u);
    if (IS_DIRECTED || (v != u))
        HGraph_qedge_to_xedge(&self->out_graph, u, v);
}

static inline void GManager_xedge_to_qedge(GManager *self, size_t v, size_t u) {
    HGraph_xedge_to_qedge(&self->inc_graph, v, u);
    if (IS_DIRECTED || (v != u))
        HGraph_xedge_to_qedge(&self->out_graph, u, v);
}
// --------------------------------- next edge ---------------------------------
bool GManager_pick_next_edge(GManager *self, edge_t *edge, size_t *vn,
                             size_t *un) {
    PROFILING_START;

    bool ret = false;

    if (IS_NODE_CLASS && USE_HEURISTIC_PICK) {
        if (GManager_pideg(self, 0) > 0) {
            *edge = PEDGE;
            *vn   = 0;
            *un   = HGraph_pmin(&self->inc_graph, 0);
            ret   = true;
            goto out;
        }

        for (size_t k = 1; k < N_LAYERS; ++k) {
            ITER_VTXS(v) {  // for compatibility
                if (!HGraph_is_gnbr(&self->inc_graph, k, v))
                    continue;
                if (IS_DEL_ONLY) {
                    if (GManager_pideg(self, v) > 0) {
                        *edge = PEDGE;
                        *vn   = v;
                        *un   = HGraph_pmin(&self->inc_graph, v);
                        ret   = true;
                        goto out;
                    }
                }
                else {
                    assert(IS_DEL_INS);
                    if (GManager_pideg(self, v) > 0) {
                        *edge = PEDGE;
                        *vn   = v;
                        *un   = HGraph_pmin(&self->inc_graph, v);
                        ret   = true;
                        goto out;
                    }
                    if (GManager_qideg(self, v) > 0) {
                        *edge = QEDGE;
                        *vn   = v;
                        *un   = HGraph_qmin(&self->inc_graph, v);
                        ret   = true;
                        goto out;
                    }
                }
            }
        }
    }
    else {
        assert((IS_NODE_CLASS && !USE_HEURISTIC_PICK) || IS_GRAPH_CLASS);
        ITER_VTXS(v) {
            ITER_VTXS(u) {
                if (IS_DEL_ONLY) {
                    if (HGraph_edge(&self->inc_graph, v, u) == PEDGE) {
                        *edge = PEDGE;
                        *vn   = v;
                        *un   = u;
                        ret   = true;
                        goto out;
                    }
                }
                else {
                    assert(IS_DEL_INS);
                    if (HGraph_edge(&self->inc_graph, v, u) == PEDGE) {
                        *edge = PEDGE;
                        *vn   = v;
                        *un   = u;
                        ret   = true;
                        goto out;
                    }
                    if (HGraph_edge(&self->inc_graph, v, u) == QEDGE) {
                        *edge = QEDGE;
                        *vn   = v;
                        *un   = u;
                        ret   = true;
                        goto out;
                    }
                }
            }
        }
    }

out:
    PROFILING_END;

    return ret;
}

// ------------------------------ edge operations ------------------------------
static void GManager_push_op(GManager *self, edge_t edge, size_t v, size_t u,
                             op_t op) {
    GManagerEdgeOp *frame = (GManagerEdgeOp *)Stack_push(&self->stack);

    frame->edge = edge;
    frame->v    = v;
    frame->u    = u;
    frame->op   = op;

    if (edge == PEDGE) {
        if (op == CUT) {
            GManager_pedge_to_xedge(self, v, u);
            BudManager_decrease(&self->budm, v, u);
        }
        else {
            assert(op == CON);
            GManager_pedge_to_nedge(self, v, u);
        }
    }
    else {
        assert(edge == QEDGE);
        if (op == CUT) {
            GManager_qedge_to_xedge(self, v, u);
        }
        else {
            assert(op == CON);
            GManager_qedge_to_nedge(self, v, u);
            BudManager_decrease(&self->budm, v, u);
        }
    }
}

// ---------------------------------- update -----------------------------------
static void GManager_set_index_b(GManager *self, uint16_t v) {
    if (self->flag_b[v])
        return;

    self->flag_b[v] = true;
    *self->end_b    = v;
    ++self->end_b;
}

static void GManager_comp_index_queue_gnbr(GManager *self, bool *flag,
                                           uint16_t *queue, uint16_t *end,
                                           uint16_t *lens) {
    lens[0]         = end - queue;
    size_t len_all  = lens[0];
    size_t len_prev = lens[0];
    for (size_t l = 1; l < N_LAYERS; ++l) {
        size_t len = 0;

        for (size_t i = 0; i < len_prev; ++i, ++queue) {
            size_t v = *queue;

            ITER_GNBRS(&self->out_graph, v, u) {
                if (flag[u])
                    continue;

                flag[u] = l;
                *end    = u;
                ++end;
                ++len;
            }
        }

        len_all += len;
        len_prev = len;

        lens[l] = len_all;
    }
}

static void GManager_comp_index_queue_anbr(GManager *self, bool *flag,
                                           uint16_t *queue, uint16_t *end,
                                           uint16_t *lens) {
    lens[0]         = end - queue;
    size_t len_all  = lens[0];
    size_t len_prev = lens[0];
    for (size_t l = 1; l < N_LAYERS; ++l) {
        size_t len = 0;

        for (size_t i = 0; i < len_prev; ++i, ++queue) {
            size_t v = *queue;

            ITER_ANBRS(&self->out_graph, v, u) {
                if (flag[u])
                    continue;

                flag[u] = l;
                *end    = u;
                ++end;
                ++len;
            }
        }

        len_all += len;
        len_prev = len;

        lens[l] = len_all;
    }
}

static void GManager_comp_index_node_gnbr(GManager *self, uint16_t *queue,
                                          uint16_t *lens, uint16_t **index,
                                          uint16_t *counter_self,
                                          uint16_t *counter_nbrs) {
    ITER_LAYERS(l) {
        uint16_t *index_l = index[l];

        uint16_t counter_self_l = 0;
        uint16_t counter_nbrs_l = 0;

        size_t len1 = lens[l - 1];
        size_t len2 = (l == 1) ? 0 : lens[l - 2];

        size_t i = 0;
        for (; i < len2; ++i) {
            size_t v = queue[i];
            if (HGraph_is_gnbr(&self->inc_graph, N_LAYERS - l, v)) {
                *index_l = v;
                index_l++;
                counter_self_l++;
                counter_nbrs_l++;
            }
        }
        for (; i < len1; ++i) {
            size_t v = queue[i];
            if (HGraph_is_gnbr(&self->inc_graph, N_LAYERS - l, v)) {
                *index_l = v;
                index_l++;
                counter_nbrs_l++;
            }
        }
        counter_self[l] = counter_self_l;
        counter_nbrs[l] = counter_nbrs_l;
    }
}

static void GManager_comp_index_node_anbr(GManager *self, uint16_t *queue,
                                          uint16_t *lens, uint16_t **index,
                                          uint16_t *counter_self,
                                          uint16_t *counter_nbrs) {
    ITER_LAYERS(l) {
        uint16_t *index_l = index[l];

        uint16_t counter_self_l = 0;
        uint16_t counter_nbrs_l = 0;

        size_t len1 = lens[l - 1];
        size_t len2 = (l == 1) ? 0 : lens[l - 2];

        size_t i = 0;
        for (; i < len2; ++i) {
            size_t v = queue[i];
            if (HGraph_is_anbr(&self->inc_graph, N_LAYERS - l, v)) {
                *index_l = v;
                index_l++;
                counter_self_l++;
                counter_nbrs_l++;
            }
        }
        for (; i < len1; ++i) {
            size_t v = queue[i];
            if (HGraph_is_anbr(&self->inc_graph, N_LAYERS - l, v)) {
                *index_l = v;
                index_l++;
                counter_nbrs_l++;
            }
        }
        counter_self[l] = counter_self_l;
        counter_nbrs[l] = counter_nbrs_l;
    }
}

void GManager_comp_index_f(GManager *self, uint16_t **index,
                           uint16_t *counter_self, uint16_t *counter_nbrs) {
    bool     *flag = self->flag_f;
    uint16_t *queue;
    uint16_t *lens;
    if (IS_NODE_CLASS) {
        queue = self->queue_f;
        lens  = self->lens_f;
    }
    else {
        assert(IS_GRAPH_CLASS);
        queue = index[1];
        lens  = counter_nbrs + 1;
    }
    uint16_t *end = queue;

    edge_t edge = self->edge;
    size_t v    = self->v;
    size_t u    = self->u;
    op_t   op   = self->op;

    memset(flag, 0, N_VERTICES * sizeof(bool));
    if (((edge == PEDGE) && (op == CUT)) || ((edge == QEDGE) && (op == CON))) {
        if (IS_DIRECTED) {
            flag[v]  = true;
            queue[0] = v;
            end++;
        }
        else {
            assert(IS_UNDIRECTED);
            flag[v]  = true;
            flag[u]  = true;
            queue[0] = v;
            queue[1] = u;
            end += 2;
        }
    }

    GManager_comp_index_queue_gnbr(self, flag, queue, end, lens);

    if (IS_NODE_CLASS)
        GManager_comp_index_node_gnbr(self, queue, lens, index, counter_self,
                                      counter_nbrs);
    else {
        assert(IS_GRAPH_CLASS);
        memcpy(counter_self + 1, counter_nbrs, N_LAYERS * sizeof(uint16_t));
        counter_self[1] = 0;
    }
}

void GManager_comp_index_b(GManager *self, uint16_t **index,
                           uint16_t *counter_self, uint16_t *counter_nbrs) {
    bool     *flag  = self->flag_b;
    uint16_t *queue = self->queue_b;
    uint16_t *end   = self->end_b;
    uint16_t *lens  = IS_NODE_CLASS ? self->lens_b : (counter_nbrs + 1);

    edge_t edge = self->edge;
    op_t   op   = self->op;

    if (((edge == PEDGE) && (op == CUT)) || ((edge == QEDGE) && (op == CON)))
        end = BudManager_update(&self->budm, flag, end);

    if (IS_DEL_ONLY)
        GManager_comp_index_queue_gnbr(self, flag, queue, end, lens);
    else {
        assert(IS_DEL_INS);
        GManager_comp_index_queue_anbr(self, flag, queue, end, lens);
    }

    if (IS_NODE_CLASS) {
        if (IS_DEL_ONLY)
            GManager_comp_index_node_gnbr(self, queue, lens, index,
                                          counter_self, counter_nbrs);
        else {
            assert(IS_DEL_INS);
            GManager_comp_index_node_anbr(self, queue, lens, index,
                                          counter_self, counter_nbrs);
        }
    }
    else {
        assert(IS_GRAPH_CLASS);
        uint16_t *tmp = index[1];
        index[1]      = self->queue_b;
        self->queue_b = tmp;

        memcpy(counter_self + 1, counter_nbrs, N_LAYERS * sizeof(uint16_t));
        counter_self[1] = 0;
    }
}

// ---------------------------------- status -----------------------------------
static void GManager_push_propagate(GManager *self, size_t v) {
    uint16_t list[N_VERTICES];
    uint16_t pdeg = GManager_pideg(self, v);
    uint16_t qdeg = GManager_qideg(self, v);

    if (pdeg > 0) {
        memcpy(list, GManager_pilist(self, v), pdeg * sizeof(uint16_t));
        ITER_UINT16_LIST(list, pdeg, u) {
            GManager_push_op(self, PEDGE, v, u, CON);
            GManager_set_index_b(self, u);
        }
    }
    if (qdeg > 0) {
        memcpy(list, GManager_qilist(self, v), qdeg * sizeof(uint16_t));
        ITER_UINT16_LIST(list, qdeg, u) {
            GManager_push_op(self, QEDGE, v, u, CUT);
            GManager_set_index_b(self, u);
        }
    }
}

void GManager_push(GManager *self, edge_t edge, size_t v, size_t u, op_t op) {
    assert(HGraph_edge(&self->inc_graph, v, u) == edge);

    PROFILING_START;

    // push
    BudManager_push(&self->budm);
    Stack_push_marker(&self->stack);

    // store operations
    self->edge = edge;
    self->v    = v;
    self->u    = u;
    self->op   = op;

    // push op and reset index_b
    GManager_push_op(self, edge, v, u, op);
    memset(self->flag_b, 0, N_VERTICES * sizeof(bool));
    if (IS_DIRECTED || (v == u)) {
        self->flag_b[v]  = true;
        self->queue_b[0] = v;
        self->end_b      = self->queue_b + 1;
    }
    else {
        assert(IS_UNDIRECTED && (v != u));
        self->flag_b[v] = true;
        self->flag_b[u] = true;

        uint16_t *queue = self->queue_b;
        queue[0]        = v;
        queue[1]        = u;
        self->end_b     = queue + 2;
    }

    // propagate if there is no local budget, but still stay within the
    // global budget
    if (GManager_gbudget(self) > 0) {
        if (IS_DIRECTED) {
            if (GManager_lbudget(self, v) == 0)
                GManager_push_propagate(self, v);
        }
        else {
            assert(IS_UNDIRECTED);
            if (GManager_lbudget(self, v) == 0)
                GManager_push_propagate(self, v);
            if (GManager_lbudget(self, u) == 0)
                GManager_push_propagate(self, u);
        }
    }

    PROFILING_END;
}

void GManager_pop(GManager *self) {
    PROFILING_START;

    BudManager_pop(&self->budm);

    GManagerEdgeOp *frame;
    while ((frame = (GManagerEdgeOp *)Stack_pop(&self->stack)) != NULL) {
        size_t v = frame->v;
        size_t u = frame->u;

        if (frame->edge == PEDGE) {
            if (frame->op == CUT)
                GManager_xedge_to_pedge(self, v, u);
            else {
                assert(frame->op == CON);
                GManager_nedge_to_pedge(self, v, u);
            }
        }
        else {
            assert(frame->edge == QEDGE);
            if (frame->op == CUT)
                GManager_xedge_to_qedge(self, v, u);
            else {
                assert(frame->op == CON);
                GManager_nedge_to_qedge(self, v, u);
            }
        }
    }

    PROFILING_END;
}
