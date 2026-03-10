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

#include "utils.h"

_Static_assert(sizeof(double) == 8, "this code assumes sizeof(double) == 8");
#define EPS       1e-5
#define N_VEC_PAD 8

// ---------------------------------- Vector -----------------------------------
typedef double *VEC;
typedef double *restrict VEC_restr;

typedef struct {
    uint16_t dim;
    uint16_t pad_dim;
} VEC_header;

#define PAD_VEC_HEADER_SIZE (PAD((sizeof(VEC_header)), ALIGNMENT))

static inline VEC_header *VEC_get_header(VEC self) {
    return (VEC_header *)((char *)self - PAD_VEC_HEADER_SIZE);
}

static inline size_t VEC_dim(VEC self) {
    return (VEC_get_header(self))->dim;
}

static inline size_t VEC_pad_dim(VEC self) {
    return (VEC_get_header(self))->pad_dim;
}

#define ITER_VEC(self, i) \
    for (size_t i = 0, dim##i = VEC_dim(self); i < dim##i; ++i)

VEC VEC_alloc(size_t dim);
VEC VEC_alloc_fp(size_t dim, FILE *fp);
void VEC_free(VEC self);
void VEC_dump(VEC self);

void VEC_resize(VEC self, size_t dim);
void VEC_shift(VEC self, size_t r, size_t shift);

void VEC_copy(VEC_restr self, VEC_restr va);
void VEC_clear(VEC self);            // self <- [0., 0., ...]
void VEC_set(VEC self, double val);  // self <- [val, val, ...]
void VEC_div(VEC self, size_t n);    // self <- self / n
void VEC_relu(VEC self);

size_t VEC_argmax(VEC self);
bool VEC_any_pos(VEC self);

// ---------------------------------- Matrix -----------------------------------
typedef VEC *MAT;
typedef VEC_restr *restrict MAT_restr;

typedef struct {
    uint16_t dim1;
    uint16_t dim2;
    uint16_t pad_dim2;

    MAT abs;
} MAT_header;

#define PAD_MAT_HEADER_SIZE (PAD((sizeof(MAT_header)), ALIGNMENT))

static inline MAT_header *MAT_get_header(MAT_restr self) {
    return (MAT_header *)((char *)self - PAD_MAT_HEADER_SIZE);
}

static inline size_t MAT_dim1(MAT_restr self) {
    return (MAT_get_header(self))->dim1;
}

static inline size_t MAT_dim2(MAT_restr self) {
    return (MAT_get_header(self))->dim2;
}

static inline size_t MAT_pad_dim2(MAT_restr self) {
    return (MAT_get_header(self))->pad_dim2;
}

static inline MAT MAT_abs(MAT_restr self) {
    return (MAT_get_header(self))->abs;
}

#define ITER_MAT1(self, i) \
    for (size_t i = 0, dim##i = MAT_dim1(self); i < dim##i; ++i)

#define ITER_MAT2(self, i) \
    for (size_t i = 0, dim##i = MAT_dim2(self); i < dim##i; ++i)

MAT MAT_alloc(size_t dim1, size_t dim2);
MAT MAT_alloc_fp(size_t dim1, size_t dim2, FILE *fp);
MAT MAT_alloc_file(size_t dim1, size_t dim2, const char *mat_path);
void MAT_free(MAT self);
void MAT_dump(MAT self);
void MAT_comp_abs(MAT self);

void MAT_resize(MAT self, size_t dim1);
void MAT_shift(MAT self, size_t r, size_t shift);

// ------------------------------- Vector-Vector -------------------------------
// vb, va <- va, vb
#define VV_SWAP(vb, va) \
    do {                \
        VEC tmp = vb;   \
        vb      = va;   \
        va      = tmp;  \
    } while (0)

void VV_add(VEC vdest, VEC va, VEC_restr vb);       // vdest <- va + vb
void VV_add_relu(VEC vdest, VEC va, VEC_restr vb);  // vdest <- relu(va + vb)
void VV_sub(VEC vdest, VEC va, VEC_restr vb);       // vdest <- va - vb
void VV_sub_pos(VEC vdest, VEC_restr va);  // vdest <- vdest - max(va, 0)
void VV_sub_neg(VEC vdest, VEC_restr va);  // vdest <- vdest - min(va, 0)

// ------------------------------- Matrix-Vector -------------------------------
// vb, ma[r] <- ma[r], vb
#define MV_SWAP(vb, ma, r) \
    do {                   \
        VEC tmp_m = ma[r]; \
        VEC tmp_v = vb;    \
        ma[r]     = tmp_v; \
        vb        = tmp_m; \
    } while (0)

// vdest <- va + sum(mb[list])
void MV_acc(VEC vdest, VEC va, MAT_restr mb, uint16_t *list, size_t len);

// vdest <- va + sum(max(mb[list], 0))
void MV_acc_pos(VEC vdest, VEC va, MAT_restr mb, uint16_t *list, size_t len);

// vdest <- va + sum(min(mb[list], 0))
void MV_acc_neg(VEC vdest, VEC va, MAT_restr mb, uint16_t *list, size_t len);

// vdest <- max(va, mb[list])
void MV_max(VEC vdest, VEC va, MAT_restr mb, uint16_t *list, size_t len);

// vdest <- min(va, mb[list])
void MV_min(VEC vdest, VEC va, MAT_restr mb, uint16_t *list, size_t len);

// ------------------------------- Matrix-Matrix -------------------------------
// vdest <- ma @ vb
void MV_dot(VEC_restr vdest, MAT_restr ma, VEC_restr vb, VEC_restr vc);

// mdest <- ma @ mb,
void MM_dot(MAT_restr mdest, MAT_restr ma, MAT_restr mb);

// mdest <- ma @ mb^T + vc (each row)
void MM_dot_trans(MAT_restr mdest, MAT_restr ma, MAT_restr mb, VEC_restr vc,
                  uint16_t *list, size_t len);

// [mdestl, mdestu] <- [mal, mau] @ mb^T + vc (each row)
void MM_relax_trans(MAT_restr mdestl, MAT_restr mdestu, MAT_restr mal,
                    MAT_restr mau, MAT_restr mb, VEC_restr vc, MAT_restr mtmps,
                    MAT_restr mtmpd, uint16_t *list, size_t len);

// ---------------------------------- sorted -----------------------------------
// vacc <- vacc + sum(ma[list]);
// vmax <- max(ma[list]);
// vmin <- min(ma[list])
// list != NULL
void MV_trinity(VEC_restr vacc, VEC_restr vmax, VEC_restr vmin, MAT_restr ma,
                uint16_t *list, size_t len);

// vdest1 >= vdest2 >= others
void MV_max2(VEC_restr vdest0, VEC_restr vdest1, MAT_restr ma, uint16_t *list,
             size_t len);

// vdest1 <= vdest2 <= others
void MV_min2(VEC_restr vdest0, VEC_restr vdest1, MAT_restr ma, uint16_t *list,
             size_t len);

// vdest1 <= vdest2 <= vdest3 <= others
void MV_min3(VEC_restr vdest0, VEC_restr vdest1, VEC_restr vdest2, MAT_restr ma,
             uint16_t *list, size_t len);
