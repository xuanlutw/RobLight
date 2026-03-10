#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "Common.h"
#include "utils.h"

typedef struct HGraph {
    uint16_t *list;
    uint16_t *pos;
    edge_t   *edge;

    uint16_t *poffset;
    uint16_t *noffset;
    uint16_t *qoffset;
    uint16_t *xoffset;

    uint16_t *pmin;
    uint16_t *qmin;

    uint16_t *gnbr_list;
    uint16_t *gnbr_lens;
    uint8_t  *gnbr_dist;

    uint16_t *anbr_list;
    uint16_t *anbr_lens;
    uint8_t  *anbr_dist;
} HGraph;

// --------------------------------- lifecycle ---------------------------------
void HGraph_init(HGraph *self);
void HGraph_init_copy(HGraph *self, HGraph *H);
void HGraph_init_copy_trans(HGraph *self, HGraph *H);
void HGraph_alias(HGraph *self, HGraph *H);
HGraph *HGraph_alloc();
HGraph *HGraph_alloc_fp(FILE *fp);
void HGraph_fill(HGraph *self);
void HGraph_cleanup(HGraph *self);
void HGraph_free(HGraph *self);

// ----------------------------- basic operations ------------------------------
static inline edge_t HGraph_edge(HGraph *self, size_t v, size_t u) {
    return self->edge[v * N_VERTICES + u];
}

static inline size_t HGraph_pdeg(HGraph *self, size_t v) {
    return self->noffset[v] - self->poffset[v];
}

static inline size_t HGraph_ndeg(HGraph *self, size_t v) {
    return self->qoffset[v] - self->noffset[v];
}

static inline size_t HGraph_qdeg(HGraph *self, size_t v) {
    return self->xoffset[v] - self->qoffset[v];
}

static inline size_t HGraph_gdeg(HGraph *self, size_t v) {
    return self->qoffset[v] - self->poffset[v];
}

static inline size_t HGraph_adeg(HGraph *self, size_t v) {
    return self->xoffset[v] - self->poffset[v];
}

static inline uint16_t *HGraph_plist(HGraph *self, size_t v) {
    return self->list + self->poffset[v];
}

static inline uint16_t *HGraph_nlist(HGraph *self, size_t v) {
    return self->list + self->noffset[v];
}

static inline uint16_t *HGraph_qlist(HGraph *self, size_t v) {
    return self->list + self->qoffset[v];
}

static inline uint16_t *HGraph_glist(HGraph *self, size_t v) {
    return self->list + self->poffset[v];
}

static inline uint16_t *HGraph_alist(HGraph *self, size_t v) {
    return self->list + self->poffset[v];
}

#define ITER_PNBRS(self, v, u) \
    ITER_UINT16_LIST(HGraph_plist(self, v), HGraph_pdeg(self, v), u)

#define ITER_QNBRS(self, v, u) \
    ITER_UINT16_LIST(HGraph_qlist(self, v), HGraph_qdeg(self, v), u)

#define ITER_GNBRS(self, v, u) \
    ITER_UINT16_LIST(HGraph_glist(self, v), HGraph_gdeg(self, v), u)

#define ITER_ANBRS(self, v, u) \
    ITER_UINT16_LIST(HGraph_alist(self, v), HGraph_adeg(self, v), u)

static inline size_t HGraph_count_edges(HGraph *self) {
    size_t counter = 0;

    ITER_VTXS(v) {
        ITER_VTXS(u) {
            if (IS_UNDIRECTED && (u > v))
                break;
            if (HGraph_edge(self, v, u) == PEDGE)
                counter++;
        }
    }

    return counter;
}

// ------------------------------ edge operations ------------------------------
static inline void HGraph_set_pedge(HGraph *self, size_t v, size_t u) {
    assert(self->noffset[v] == self->xoffset[v]);

    size_t index = v * N_VERTICES + u;
    assert(self->edge[index] == XEDGE);

    size_t pos        = self->noffset[v];
    self->list[pos]   = u;
    self->pos[index]  = pos;
    self->edge[index] = PEDGE;

    ++self->noffset[v];
    ++self->qoffset[v];
    ++self->xoffset[v];

    if (self->pmin[v] != 0xFFFF)
        self->pmin[v] = MIN(self->pmin[v], u);
}

static inline void HGraph_set_qedge(HGraph *self, size_t v, size_t u) {
    size_t index = v * N_VERTICES + u;
    assert(self->edge[index] == XEDGE);

    size_t pos        = self->xoffset[v];
    self->list[pos]   = u;
    self->pos[index]  = pos;
    self->edge[index] = QEDGE;

    ++self->xoffset[v];

    if (self->qmin[v] != 0xFFFF)
        self->qmin[v] = MIN(self->qmin[v], u);
}

static inline void HGraph_exchange(HGraph *self, size_t v, size_t u,
                                   size_t pos_w) {
    size_t w = self->list[pos_w];

    size_t index_u = v * N_VERTICES + u;
    size_t index_w = v * N_VERTICES + w;

    size_t pos_u = self->pos[index_u];

    // set u
    self->list[pos_w]  = u;
    self->pos[index_u] = pos_w;

    // set w
    self->list[pos_u]  = w;
    self->pos[index_w] = pos_u;
}

static inline void HGraph_move(HGraph *self, size_t v, size_t u, size_t pos_w) {
    size_t index_u = v * N_VERTICES + u;

    // set u
    self->list[pos_w]  = u;
    self->pos[index_u] = pos_w;
}

static inline void HGraph_nedge_to_pedge(HGraph *self, size_t v, size_t u) {
    size_t index_u = v * N_VERTICES + u;
    assert(self->edge[index_u] == NEDGE);

    HGraph_exchange(self, v, u, self->noffset[v]);

    self->edge[index_u] = PEDGE;
    ++self->noffset[v];

    if (self->pmin[v] != 0xFFFF)
        self->pmin[v] = MIN(self->pmin[v], u);
}

static inline void HGraph_pedge_to_nedge(HGraph *self, size_t v, size_t u) {
    size_t index_u = v * N_VERTICES + u;
    assert(self->edge[index_u] == PEDGE);

    HGraph_exchange(self, v, u, self->noffset[v] - 1);

    self->edge[index_u] = NEDGE;
    --self->noffset[v];

    if (self->pmin[v] == u)
        self->pmin[v] = 0xFFFF;
}

static inline void HGraph_xedge_to_pedge(HGraph *self, size_t v, size_t u) {
    size_t index_u = v * N_VERTICES + u;
    assert(self->edge[index_u] == XEDGE);

    HGraph_move(self, v, u, self->poffset[v] - 1);

    self->edge[index_u] = PEDGE;
    --self->poffset[v];

    if (self->pmin[v] != 0xFFFF)
        self->pmin[v] = MIN(self->pmin[v], u);
}

static inline void HGraph_pedge_to_xedge(HGraph *self, size_t v, size_t u) {
    size_t index_u = v * N_VERTICES + u;
    assert(self->edge[index_u] == PEDGE);

    size_t pos_u = self->pos[index_u];
    size_t pos_w = self->poffset[v];
    size_t w     = self->list[pos_w];

    HGraph_move(self, v, w, pos_u);

    self->edge[index_u] = XEDGE;
    ++self->poffset[v];

    if (self->pmin[v] == u)
        self->pmin[v] = 0xFFFF;
}

static inline void HGraph_nedge_to_qedge(HGraph *self, size_t v, size_t u) {
    size_t index_u = v * N_VERTICES + u;
    assert(self->edge[index_u] == NEDGE);

    HGraph_exchange(self, v, u, self->qoffset[v] - 1);

    self->edge[index_u] = QEDGE;
    --self->qoffset[v];

    if (self->qmin[v] != 0xFFFF)
        self->qmin[v] = MIN(self->qmin[v], u);
}

static inline void HGraph_qedge_to_nedge(HGraph *self, size_t v, size_t u) {
    size_t index_u = v * N_VERTICES + u;
    assert(self->edge[index_u] == QEDGE);

    HGraph_exchange(self, v, u, self->qoffset[v]);

    self->edge[index_u] = NEDGE;
    ++self->qoffset[v];

    if (self->qmin[v] == u)
        self->qmin[v] = 0xFFFF;
}

static inline void HGraph_xedge_to_qedge(HGraph *self, size_t v, size_t u) {
    size_t index_u = v * N_VERTICES + u;
    assert(self->edge[index_u] == XEDGE);

    HGraph_move(self, v, u, self->xoffset[v]);

    self->edge[index_u] = QEDGE;
    ++self->xoffset[v];

    if (self->qmin[v] != 0xFFFF)
        self->qmin[v] = MIN(self->qmin[v], u);
}

static inline void HGraph_qedge_to_xedge(HGraph *self, size_t v, size_t u) {
    size_t index_u = v * N_VERTICES + u;
    assert(self->edge[index_u] == QEDGE);

    size_t pos_u = self->pos[index_u];
    size_t pos_w = self->xoffset[v] - 1;
    size_t w     = self->list[pos_w];

    HGraph_move(self, v, w, pos_u);

    self->edge[index_u] = XEDGE;
    --self->xoffset[v];

    if (self->qmin[v] == u)
        self->qmin[v] = 0xFFFF;
}

// ------------------------------------ min ------------------------------------
static inline uint16_t HGraph_pmin(HGraph *self, size_t v) {
    if (self->pmin[v] == 0xFFFF) {
        size_t min = 0xFFFF;
        ITER_PNBRS(self, v, u) {
            min = MIN(min, u);
        }
        self->pmin[v] = min;
    }
    return self->pmin[v];
}

static inline uint16_t HGraph_qmin(HGraph *self, size_t v) {
    if (self->qmin[v] == 0xFFFF) {
        size_t min = 0xFFFF;
        ITER_PNBRS(self, v, u) {
            min = MIN(min, u);
        }
        self->qmin[v] = min;
    }
    return self->qmin[v];
}

// ------------------------------ k-hop neighbors ------------------------------
static inline bool HGraph_is_gnbr(HGraph *self, size_t k, size_t u) {
    assert(k <= N_LAYERS);

    return (self->gnbr_dist[u] <= k);
}

static inline bool HGraph_is_anbr(HGraph *self, size_t k, size_t u) {
    assert(k <= N_LAYERS);

    return (self->anbr_dist[u] <= k);
}

static inline uint16_t *HGraph_gnbr_list(HGraph *self) {
    return self->gnbr_list;
}

static inline uint16_t *HGraph_anbr_list(HGraph *self) {
    return self->anbr_list;
}

static inline size_t HGraph_gnbr_len(HGraph *self, size_t k) {
    return self->gnbr_lens[k];
}

static inline size_t HGraph_anbr_len(HGraph *self, size_t k) {
    return self->anbr_lens[k];
}

void HGraph_update_gnbr(HGraph *self);
void HGraph_update_anbr(HGraph *self);
