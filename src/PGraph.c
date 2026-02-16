#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "BArray.h"
#include "BudManager.h"
#include "Common.h"
#include "Graph.h"
#include "PGraph.h"
#include "Stack.h"
#include "utils.h"

// ----------------------------------- type ------------------------------------
typedef struct {
    Node node;

    Edge_type edge_type;
    size_t    v;
    size_t    u;
    Op_type   op_type;
} PGraphEdgeOp;

// --------------------------------- liftcycle ---------------------------------
PGraph *PGraph_alloc(Common *common) {
    PGraph *self = XMALLOC(sizeof(PGraph));
    self->common = common;

    self->GP = Graph_alloc(common);
    self->GQ = Graph_alloc(common);
    self->GN = Graph_alloc(common);
    self->G  = Graph_alloc(common);

    self->updated = BArray_alloc(N_VERTICES);

    self->budm = BudManager_alloc(common);

    self->stack = Stack_alloc(sizeof(PGraphEdgeOp), 1, NULL);

    self->clock = 0;

    return self;
}

void PGraph_free(PGraph *self) {
    Graph_free(self->GP);
    Graph_free(self->GQ);
    Graph_free(self->GN);
    Graph_free(self->G);

    BArray_free(self->updated);

    BudManager_free(self->budm);

    Stack_free(self->stack);

    free(self);
}

void PGraph_dump(PGraph *self) {
    printf("#vertices: %ld\n", N_VERTICES);

    printf("========== potential edges ==========\n");
    Graph_dump(self->GP, false);
    if (IS_DEL_INS) {
        printf("========== (Q)potential edges ==========\n");
        Graph_dump(self->GQ, false);
    }
    printf("========== normal edges ==========\n");
    Graph_dump(self->GN, false);
    printf("========== all edges ==========\n");
    Graph_dump(self->G, true);
    printf("next edge: (%lu, %lu)\n", self->v, self->u);
    printf("========== budgets ==========\n");
    BudManager_dump(self->budm);
}

// --------------------------------- next edge ---------------------------------
bool PGraph_pick_next_edge(PGraph *self) {
    PROFILING_START;

    // comp next edge
    bool has_next = false;

    if (IS_NODE_CLASS && USE_HEURISTIC_PICK) {
        if (Graph_ideg(self->GP, 0) > 0) {
            self->edge_type = PEDGE;
            self->v         = 0;
            self->u         = (Graph_ilist(self->GP, 0))[0];
            has_next        = true;
        }

        for (size_t k = 1; k < N_LAYERS; ++k) {
            if (has_next) {
                if ((self->v == 0) && (self->u == 0))
                    return false;
                break;
            }
            ITER_KNBRS(self, k, v) {
                if (IS_DEL_ONLY) {
                    if (Graph_ideg(self->GP, v) > 0) {
                        self->edge_type = PEDGE;
                        self->v         = v;
                        self->u         = (Graph_ilist(self->GP, v))[0];
                        has_next        = true;
                        break;
                    }
                }
                if (IS_DEL_INS) {
                    if (Graph_ideg(self->GP, v) > 0) {
                        self->edge_type = PEDGE;
                        self->v         = v;
                        self->u         = (Graph_ilist(self->GP, v))[0];
                        has_next        = true;
                        break;
                    }
                    if (Graph_ideg(self->GQ, v) > 0) {
                        self->edge_type = QEDGE;
                        self->v         = v;
                        self->u         = (Graph_ilist(self->GQ, v))[0];
                        has_next        = true;
                        break;
                    }
                }
            }
        }
    }
    if ((IS_NODE_CLASS && !USE_HEURISTIC_PICK) || IS_GRAPH_CLASS) {
        ITER_VTXS(v) {
            if (has_next)
                break;
            ITER_VTXS(u) {
                if (IS_DEL_ONLY) {
                    if (Graph_is_edge(self->GP, v, u)) {
                        self->edge_type = PEDGE;
                        self->v         = v;
                        self->u         = u;
                        has_next        = true;
                        break;
                    }
                }
                if (IS_DEL_INS) {
                    if (Graph_is_edge(self->GP, v, u)) {
                        self->edge_type = PEDGE;
                        self->v         = v;
                        self->u         = u;
                        has_next        = true;
                        break;
                    }
                    if (Graph_is_edge(self->GQ, v, u)) {
                        self->edge_type = QEDGE;
                        self->v         = v;
                        self->u         = u;
                        has_next        = true;
                        break;
                    }
                }
            }
        }
    }

    PROFILING_END;

    return has_next;
}

// ------------------------------ edge operations ------------------------------
static void PGraph_push_op(PGraph *self, Edge_type edge_type, size_t v,
                           size_t u, Op_type op_type) {
    PGraphEdgeOp *op = (PGraphEdgeOp *)Stack_push(self->stack, 1);

    op->edge_type = edge_type;
    op->v         = v;
    op->u         = u;
    op->op_type   = op_type;

    if (edge_type == PEDGE) {
        if (op_type == CUT) {
            Graph_del_edge(self->GP, v, u);
            Graph_del_edge(self->G, v, u);
            BudManager_decrease(self->budm, v, u);
        }
        if (op_type == CON) {
            Graph_del_edge(self->GP, v, u);
            Graph_set_edge(self->GN, v, u);
        }
    }
    if (edge_type == QEDGE) {
        if (op_type == CUT) {
            Graph_del_edge(self->GQ, v, u);
        }
        if (op_type == CON) {
            Graph_del_edge(self->GQ, v, u);
            Graph_set_edge(self->GN, v, u);
            Graph_set_edge(self->G, v, u);
            BudManager_decrease(self->budm, v, u);
        }
    }
}

// ---------------------------------- status -----------------------------------
void PGraph_init(PGraph *self, size_t gbudget, size_t lbudget) {
    Graph_copy(self->GP, INPUT_G);
    Graph_copy(self->G, INPUT_G);
    if (IS_DEL_INS) {
        ITER_VTXS(v) {
            ITER_VTXS(u) {
                if (v == u)
                    continue;
                if (!Graph_is_edge(self->GP, v, u))
                    Graph_set_edge(self->GQ, v, u);
            }
        }
    }

    if (IS_NODE_CLASS) {
        Graph_update_knbr(self->G);
    }

    BudManager_init(self->budm, gbudget, lbudget);
    BArray_clear(self->updated);
}

static void PGraph_push_propagate(PGraph *self, size_t v) {
    if (IS_DEL_ONLY) {
        if (PGraph_pideg(self, v) > 0) {
            ITER_PINBRS(self, v, w) {
                PGraph_push_op(self, PEDGE, v, w, CON);
            }
            BArray_union(self->updated, Graph_iadj(self->GP, v));
        }
    }
    if (IS_DEL_INS) {
        if (PGraph_pideg(self, v) > 0) {
            ITER_PINBRS(self, v, w) {
                PGraph_push_op(self, PEDGE, v, w, CON);
            }
            BArray_union(self->updated, Graph_iadj(self->GP, v));
        }
        if (PGraph_qideg(self, v) > 0) {
            ITER_QINBRS(self, v, w) {
                PGraph_push_op(self, QEDGE, v, w, CUT);
            }
            BArray_union(self->updated, Graph_iadj(self->GQ, v));
        }
    }
}

void PGraph_push(PGraph *self, Edge_type edge_type, size_t v, size_t u,
                 Op_type op_type) {
    assert((edge_type != PEDGE) || Graph_is_edge(self->GP, v, u));
    assert((edge_type != QEDGE) || Graph_is_edge(self->GQ, v, u));

    PROFILING_START;

    // push
    BudManager_push(self->budm);
    BArray_clear(self->updated);
    Stack_push_marker(self->stack);

    // store operations
    self->edge_type = edge_type;
    self->v         = v;
    self->u         = u;
    self->op_type   = op_type;

    // push op and set updated
    PGraph_push_op(self, edge_type, v, u, op_type);
    if (IS_DIRECTED) {
        BArray_set(self->updated, v);
    }
    if (IS_UNDIRECTED) {
        BArray_set(self->updated, v);
        BArray_set(self->updated, u);
    }

    // propagate if no local budget
    if (IS_DIRECTED) {
        if (PGraph_lbudget(self, v) == 0)
            PGraph_push_propagate(self, v);
    }
    if (IS_UNDIRECTED) {
        if (PGraph_lbudget(self, v) == 0)
            PGraph_push_propagate(self, v);
        if (PGraph_lbudget(self, u) == 0)
            PGraph_push_propagate(self, u);
    }

    if (IS_NODE_CLASS)
        Graph_update_knbr(self->G);

    PROFILING_END;
}

void PGraph_pop(PGraph *self) {
    PROFILING_START;

    BudManager_pop(self->budm);

    PGraphEdgeOp *op;
    while ((op = (PGraphEdgeOp *)Stack_pop(self->stack)) != NULL) {
        if (op->edge_type == PEDGE) {
            if (op->op_type == CUT) {
                Graph_set_edge(self->GP, op->v, op->u);
                Graph_set_edge(self->G, op->v, op->u);
            }
            if (op->op_type == CON) {
                Graph_set_edge(self->GP, op->v, op->u);
                Graph_del_edge(self->GN, op->v, op->u);
            }
        }
        if (op->edge_type == QEDGE) {
            if (op->op_type == CUT) {
                Graph_set_edge(self->GQ, op->v, op->u);
            }
            if (op->op_type == CON) {
                Graph_set_edge(self->GQ, op->v, op->u);
                Graph_del_edge(self->GN, op->v, op->u);
                Graph_del_edge(self->G, op->v, op->u);
            }
        }
    }

    PROFILING_END;
}
