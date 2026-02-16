#pragma once

#include <assert.h>
#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <immintrin.h>

#include "BArray.h"

_Static_assert(sizeof(double) == 8, "this code assumes sizeof(double) == 8");
#define ALIGNMENT 64 // fit cache line
#define EPS       1e-5
#define N_VEC_PAD 16

// ---------------------------------- Vector -----------------------------------
typedef struct {
    size_t  dim;
    size_t  padded_dim;
    double *data;
} VEC;

#define ITER_VEC(self, i) \
    for (size_t i = 0; i < self->dim; ++i)

#define VVAL(self, i) \
    (self->data[i])

VEC *VEC_alloc(size_t dim);
VEC *VEC_alloc_fp(size_t dim, FILE *fp);
void VEC_free(VEC *self);
void VEC_dump(VEC *self);
void VEC_copy(VEC *restrict self, const VEC *restrict va);

void VEC_resize(VEC *self, size_t dim);
void VEC_shift(VEC *self, size_t r, size_t shift);

void VEC_relu(VEC *self);
void VEC_set(VEC *self, double val);  // self <- [val, val, ...]
void VEC_clear(VEC *self);            // self <- [0., 0., ...]
void VEC_div(VEC *self, size_t n);    // self <- self / n

size_t VEC_argmax(const VEC *self);
bool VEC_any_pos(const VEC *self);

// ---------------------------------- Matrix -----------------------------------
typedef struct {
    size_t   dim1;
    size_t   dim2;
    double **data;
    VEC    **rows;
} MAT;

#define ITER_MAT1(self, i) \
    for (size_t i = 0; i < self->dim1; ++i)

#define ITER_MAT2(self, i) \
    for (size_t i = 0; i < self->dim2; ++i)

#define MVAL(self, i, j) \
    (self->data[i][j])

#define MSLICE(self, i) \
    (self->rows[i])

MAT *MAT_alloc(size_t dim1, size_t dim2);
MAT *MAT_alloc_fp(size_t dim1, size_t dim2, FILE *fp);
MAT *MAT_alloc_file(size_t dim1, size_t dim2, const char *mat_path);
void MAT_free(MAT *self);
void MAT_dump(MAT *self);

void MAT_resize(MAT *self, size_t dim1);
void MAT_shift(MAT *self, size_t r, size_t shift);

// ------------------------------- Vector-Vector -------------------------------
void VV_add(VEC *vdest, const VEC *va, const VEC *vb);  // vdest <- va + vb
void VV_sub(VEC *vdest, const VEC *va, const VEC *vb);  // vdest <- va - vb
void VV_sub_pos(VEC *vdest, const VEC *va);  // vdest <- vdest - max(va, 0)
void VV_sub_neg(VEC *vdest, const VEC *va);  // vdest <- vdest - min(va, 0)

// ------------------------------- Matrix-Vector -------------------------------
// vb, ma[r] <- ma[r], vb
#define MV_SWAP(vb, ma, r)               \
    do {                                 \
        double *tmp_vb    = vb->data;    \
        double *tmp_ma    = ma->data[r]; \
        vb->data          = tmp_ma;      \
        ma->data[r]       = tmp_vb;      \
        ma->rows[r]->data = tmp_vb;      \
    } while(0)

// vdest <- va + sum(mb[index])
void MV_acc(VEC *vdest, const VEC *va, const MAT *restrict mb, BArray *ba);

// vdest <- va + sum(max(mb[index], 0))
void MV_acc_pos(VEC *vdest, const VEC *va, const MAT *restrict mb, BArray *ba);

// vdest <- va + sum(min(mb[index], 0))
void MV_acc_neg(VEC *vdest, const VEC *va, const MAT *restrict mb, BArray *ba);

// vdest <- max(va, mb[index])
void MV_max(VEC *vdest, const VEC *va, const MAT *restrict mb, BArray *ba);

// vdest <- min(va, mb[index])
void MV_min(VEC *vdest, const VEC *va, const MAT *restrict mb, BArray *ba);

// vdest <- ma @ vb
void MV_dot(VEC *restrict vdest, const MAT *restrict ma,
            const VEC *restrict vb);

// [vdestl, vdestu] <- ma @ [vbl, vbu]
void MV_relax(VEC *restrict vdestl, VEC *restrict vdestu,
              const MAT *restrict ma, const MAT *restrict ma_abs,
              const VEC *restrict vbl, const VEC *restrict vbu,
              VEC *restrict tmps, VEC *restrict tmpd);

// ------------------------------- Matrix-Matrix -------------------------------
// mdest <- ma @ mb,
void MM_dot(MAT *restrict mdest, const MAT *restrict ma,
            const MAT *restrict mb);

// mdest <- ma @ mb^T,
void MM_dot_trans(MAT *restrict mdest, const MAT *restrict ma,
                  const MAT *restrict mb);

// ---------------------------------- sorted -----------------------------------
// vacc <- vacc + sum(ma[index]);
// vmax <- max(ma[index]);
// vmin <- min(ma[index])
// ba != NULL
void MV_trinity(VEC *restrict vacc, VEC *restrict vmax, VEC *restrict vmin,
                const MAT *restrict ma, BArray *ba);

// vdest1, vdest2 <- max2(ma[index])
void MV_max2(VEC *restrict vdest1, VEC *restrict vdest2, const MAT *restrict ma,
             BArray *ba);

// vdest1, vdest2 <- min2(ma[index])
void MV_min2(VEC *restrict vdest1, VEC *restrict vdest2, const MAT *restrict ma,
             BArray *ba);
