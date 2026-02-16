#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "BArray.h"
#include "Common.h"
#include "Graph.h"
#include "utils.h"

// --------------------------------- liftcycle ---------------------------------
Graph *Graph_alloc(Common *common) {
    Graph *self = XMALLOC(sizeof(Graph));
    assert(self != NULL);

    self->common = common;

    if (IS_DIRECTED) {
        self->iadj = XMALLOC(N_VERTICES * sizeof(BArray *));
        self->oadj = XMALLOC(N_VERTICES * sizeof(BArray *));
        ITER_VTXS(v) {
            self->iadj[v] = BArray_alloc(N_VERTICES);
            self->oadj[v] = BArray_alloc(N_VERTICES);
        }
    }
    if (IS_UNDIRECTED) {
        self->iadj = XMALLOC(N_VERTICES * sizeof(BArray *));
        self->oadj = self->iadj;
        ITER_VTXS(v) {
            self->iadj[v] = BArray_alloc(N_VERTICES);
        }
    }

    self->kadj = XMALLOC(N_LAYERS_EXT * sizeof(BArray *));
    ITER_LAYERS_EXT(k) {
        self->kadj[k] = BArray_alloc(N_VERTICES);
    }
    BArray_set(self->kadj[0], 0);  // set 0-hop neighbors

    return self;
}

Graph *Graph_alloc_fp(Common *common, FILE *fp) {
    Graph *self = Graph_alloc(common);

    ITER_VTXS(v) {
        size_t deg, u;
        fscanf(fp, "%lu", &deg);
        for (size_t i = 0; i < deg; ++i) {
            fscanf(fp, "%lu", &u);
            Graph_set_edge(self, v, u);
        }
        assert(deg == Graph_ideg(self, v));
    }

    return self;
}

void Graph_free(Graph *self) {
    if (IS_DIRECTED) {
        ITER_VTXS(v) {
            BArray_free(self->iadj[v]);
            BArray_free(self->oadj[v]);
        }
        free(self->iadj);
        free(self->oadj);
    }
    if (IS_UNDIRECTED) {
        ITER_VTXS(v) {
            BArray_free(self->iadj[v]);
        }
        free(self->iadj);
    }

    ITER_LAYERS_EXT(k) {
        BArray_free(self->kadj[k]);
    }
    free(self->kadj);

    free(self);
}

void Graph_dump(Graph *self, bool with_knbr) {
    printf("  ");
    ITER_VTXS(u) {
        printf("%2ld", u);
    }
    printf("\n");

    ITER_VTXS(v) {
        printf("%2ld", v);
        ITER_VTXS(u) {
            printf(" %c", Graph_is_edge(self, v, u) ? '+' : ' ');
        }
        printf("%2ld\n", Graph_ideg(self, v));
    }

    printf("  ");
    ITER_VTXS(u) {
        printf("%2ld", Graph_odeg(self, u));
    }
    printf("\n");

    if (with_knbr) {
        Graph_update_knbr(self);
        ITER_LAYERS_EXT(k) {
            printf("%2ld", k);
            ITER_VTXS(u) {
                printf(" %c", Graph_is_knbr(self, k, u) ? '+' : ' ');
            }
            printf("%2ld", Graph_n_knbr(self, k));
            printf("\n");
        }
    }
}

void Graph_copy(Graph *self, Graph *G) {
    if (IS_DIRECTED) {
        ITER_VTXS(v) {
            BArray_copy(self->iadj[v], G->iadj[v]);
            BArray_copy(self->oadj[v], G->oadj[v]);
        }
    }
    if (IS_UNDIRECTED) {
        ITER_VTXS(v) {
            BArray_copy(self->iadj[v], G->iadj[v]);
        }
    }
}

// ----------------------------- basic operations ------------------------------
size_t Graph_count_edges(Graph *self) {
    size_t counter = 0;

    ITER_VTXS(v) {
        ITER_VTXS(u) {
            if (IS_UNDIRECTED && (u > v))
                break;
            if (Graph_is_edge(self, v, u))
                counter++;
        }
    }

    return counter;
}

// ------------------------- k-hop incoming neighbors --------------------------
void Graph_update_knbr(Graph *self) {
    ITER_LAYERS_EXT(k) {
        if (k == 0) {
            continue;
        }
        else if (k == 1) {
            BArray_copy(self->kadj[1], self->iadj[0]);
            BArray_set(self->kadj[1], 0);
        }
        else {
            BArray_copy(self->kadj[k], self->kadj[k - 1]);
            ITER_BArray(self->kadj[k - 1], v) {
                if (BArray_test(self->kadj[k - 2], v))
                    continue;
                BArray_union(self->kadj[k], self->iadj[v]);
            }
        }
    }
}
