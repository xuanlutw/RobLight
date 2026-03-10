#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Common.h"
#include "Graph.h"
#include "utils.h"

// --------------------------------- liftcycle ---------------------------------
void HGraph_init(HGraph *self) {
    self->list = XMALLOC(N_VERTICES * N_VERTICES * sizeof(uint16_t));
    self->pos  = XMALLOC(N_VERTICES * N_VERTICES * sizeof(uint16_t));
    self->edge = XMALLOC(N_VERTICES * N_VERTICES * sizeof(edge_t));

    self->poffset = XMALLOC(N_VERTICES * sizeof(uint16_t));
    self->noffset = XMALLOC(N_VERTICES * sizeof(uint16_t));
    self->qoffset = XMALLOC(N_VERTICES * sizeof(uint16_t));
    self->xoffset = XMALLOC(N_VERTICES * sizeof(uint16_t));

    self->pmin = XMALLOC(N_VERTICES * sizeof(uint16_t));
    self->qmin = XMALLOC(N_VERTICES * sizeof(uint16_t));

    self->gnbr_list = XMALLOC(N_VERTICES * sizeof(uint16_t));
    self->gnbr_lens = XMALLOC(N_LAYERS * sizeof(uint16_t));
    self->gnbr_dist = XMALLOC(N_VERTICES * sizeof(uint8_t));

    self->anbr_list = XMALLOC(N_VERTICES * sizeof(uint16_t));
    self->anbr_lens = XMALLOC(N_LAYERS * sizeof(uint16_t));
    self->anbr_dist = XMALLOC(N_VERTICES * sizeof(uint8_t));

    ITER_VTXS(v) {
        ITER_VTXS(u) {
            self->edge[v * N_VERTICES + u] = XEDGE;
        }
        self->poffset[v] = N_VERTICES * v;
        self->noffset[v] = N_VERTICES * v;
        self->qoffset[v] = N_VERTICES * v;
        self->xoffset[v] = N_VERTICES * v;
    }

    memset(self->pmin, -1, N_VERTICES * sizeof(uint16_t));
    memset(self->qmin, -1, N_VERTICES * sizeof(uint16_t));
}

void HGraph_init_copy(HGraph *self, HGraph *H) {
    HGraph_init(self);

    memcpy(self->list, H->list, N_VERTICES * N_VERTICES * sizeof(uint16_t));
    memcpy(self->pos, H->pos, N_VERTICES * N_VERTICES * sizeof(uint16_t));
    memcpy(self->edge, H->edge, N_VERTICES * N_VERTICES * sizeof(edge_t));

    memcpy(self->poffset, H->poffset, N_VERTICES * sizeof(uint16_t));
    memcpy(self->noffset, H->noffset, N_VERTICES * sizeof(uint16_t));
    memcpy(self->qoffset, H->qoffset, N_VERTICES * sizeof(uint16_t));
    memcpy(self->xoffset, H->xoffset, N_VERTICES * sizeof(uint16_t));

    memcpy(self->pmin, H->pmin, N_VERTICES * sizeof(uint16_t));
    memcpy(self->qmin, H->qmin, N_VERTICES * sizeof(uint16_t));
}

void HGraph_init_copy_trans(HGraph *self, HGraph *H) {
    HGraph_init(self);

    // set pedges
    ITER_VTXS(v) {
        ITER_VTXS(u) {
            if (HGraph_edge(H, u, v) == PEDGE)
                HGraph_set_pedge(self, v, u);
        }
    }

    // set qedges
    ITER_VTXS(v) {
        ITER_VTXS(u) {
            if (HGraph_edge(H, u, v) == QEDGE)
                HGraph_set_qedge(self, v, u);
        }
    }
}

void HGraph_alias(HGraph *self, HGraph *H) {
    self->list = H->list;
    self->pos  = H->pos;
    self->edge = H->edge;

    self->poffset = H->poffset;
    self->noffset = H->noffset;
    self->qoffset = H->qoffset;
    self->xoffset = H->xoffset;

    self->pmin = H->pmin;
    self->qmin = H->qmin;
}

HGraph *HGraph_alloc() {
    HGraph *self = XMALLOC(sizeof(HGraph));

    HGraph_init(self);

    return self;
}

HGraph *HGraph_alloc_fp(FILE *fp) {
    HGraph *self = HGraph_alloc();

    ITER_VTXS(v) {
        size_t deg, u;
        fscanf(fp, "%lu", &deg);
        for (size_t i = 0; i < deg; ++i) {
            fscanf(fp, "%lu", &u);
            HGraph_set_pedge(self, v, u);
        }
        assert(deg == HGraph_pdeg(self, v));
    }

    return self;
}

void HGraph_fill(HGraph *self) {
    ITER_VTXS(v) {
        ITER_VTXS(u) {
            if (v == u)
                continue;
            if (HGraph_edge(self, v, u) == XEDGE)
                HGraph_set_qedge(self, v, u);
        }
    }
}

void HGraph_cleanup(HGraph *self) {
    free(self->list);
    free(self->pos);
    free(self->edge);

    free(self->poffset);
    free(self->noffset);
    free(self->qoffset);
    free(self->xoffset);

    free(self->pmin);
    free(self->qmin);

    free(self->gnbr_list);
    free(self->gnbr_lens);
    free(self->gnbr_dist);

    free(self->anbr_list);
    free(self->anbr_lens);
    free(self->anbr_dist);
}

void HGraph_free(HGraph *self) {
    HGraph_cleanup(self);

    free(self);
}

// ------------------------- k-hop grounding neighbors -------------------------
void HGraph_update_gnbr(HGraph *self) {
    uint16_t *queue = self->gnbr_list;
    uint16_t *end   = self->gnbr_list + 1;
    uint16_t *lens  = self->gnbr_lens;

    uint8_t *dist = self->gnbr_dist;
    memset(dist, -1, N_VERTICES * sizeof(uint8_t));

    dist[0]         = 0;
    queue[0]        = 0;
    lens[0]         = 1;
    size_t len_all  = 1;
    size_t len_prev = 1;

    for (size_t l = 1; l < N_LAYERS; ++l) {
        size_t len = 0;

        for (size_t i = 0; i < len_prev; ++i, ++queue) {
            size_t v = *queue;

            ITER_GNBRS(self, v, u) {
                if (dist[u] < 0xFF)
                    continue;
                dist[u] = l;
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

void HGraph_update_anbr(HGraph *self) {
    uint16_t *queue = self->anbr_list;
    uint16_t *end   = self->anbr_list + 1;
    uint16_t *lens  = self->anbr_lens;

    uint8_t *dist = self->anbr_dist;
    memset(dist, -1, N_VERTICES * sizeof(uint8_t));

    dist[0]         = 0;
    queue[0]        = 0;
    lens[0]         = 1;
    size_t len_all  = 1;
    size_t len_prev = 1;

    for (size_t l = 1; l < N_LAYERS; ++l) {
        size_t len = 0;

        for (size_t i = 0; i < len_prev; ++i, ++queue) {
            size_t v = *queue;

            ITER_ANBRS(self, v, u) {
                if (dist[u] < 0xFF)
                    continue;
                dist[u] = l;
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
