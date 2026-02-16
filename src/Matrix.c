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
#include "Matrix.h"
#include "utils.h"

// -----------------------------------------------------------------------------
#define LOAD_CONST_by_16(var, val)  \
    __m256d var[4];                 \
    var[0] = _mm256_set1_pd(val);   \
    var[1] = _mm256_set1_pd(val);   \
    var[2] = _mm256_set1_pd(val);   \
    var[3] = _mm256_set1_pd(val);

static __m256d hsum(__m256d y[4]) {
    __m256d t0 = _mm256_unpacklo_pd(y[0], y[1]);
    __m256d t1 = _mm256_unpackhi_pd(y[0], y[1]);
    __m256d t2 = _mm256_unpacklo_pd(y[2], y[3]);
    __m256d t3 = _mm256_unpackhi_pd(y[2], y[3]);

    __m256d u0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    __m256d u1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    __m256d u2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    __m256d u3 = _mm256_permute2f128_pd(t1, t3, 0x31);

    __m256d sum0 = _mm256_add_pd(u0, u1);
    __m256d sum1 = _mm256_add_pd(u2, u3);

    return _mm256_add_pd(sum0, sum1);
}

#define CSWAP(a, b)            \
    tmp = _mm256_max_pd(a, b); \
    b   = _mm256_min_pd(a, b); \
    a   = tmp

// ---------------------------------- Vector -----------------------------------
#define ITER_VEC_by_16(self, i) \
    for (size_t i = 0; i + 15 < self->padded_dim; i += 16)

#define LOAD_VEC(self, offset) \
    _mm256_load_pd(self->data + (offset))

#define LOAD_VEC_by_16(var, self, offset)  \
    __m256d var[4];                        \
    var[0] = LOAD_VEC(self, (offset));     \
    var[1] = LOAD_VEC(self, (offset) + 4); \
    var[2] = LOAD_VEC(self, (offset) + 8); \
    var[3] = LOAD_VEC(self, (offset) + 12)

#define STORE_VEC(self, offset, var) \
    _mm256_store_pd(self->data + (offset), var)

#define STORE_VEC_by_16(self, offset, var)  \
    STORE_VEC(self, (offset),      var[0]); \
    STORE_VEC(self, (offset) + 4,  var[1]); \
    STORE_VEC(self, (offset) + 8,  var[2]); \
    STORE_VEC(self, (offset) + 12, var[3])

#define VV_FIT(v1, v2) \
    assert(v1->dim == v2->dim)

VEC *VEC_alloc(size_t dim) {
    VEC *self = XMALLOC(sizeof(VEC));

    self->dim        = dim;
    self->padded_dim = PAD(dim, N_VEC_PAD);
    self->data = aligned_alloc(ALIGNMENT, self->padded_dim * sizeof(double));
    CHECK(self->data == NULL, "aligned_alloc failed!");

    memset(self->data, 0, self->padded_dim * sizeof(double));

    return self;
}

VEC *VEC_alloc_fp(size_t dim, FILE *fp) {
    VEC *self = VEC_alloc(dim);

    ITER_VEC(self, i) {
        fscanf(fp, "%lf", &VVAL(self, i));
    }

    return self;
}

void VEC_free(VEC *self) {
    free(self->data);
    free(self);
}

void VEC_dump(VEC *self) {
    ITER_VEC(self, i) {
        printf("%6.2lf ", VVAL(self, i));
    }
    printf("\n");
}

void VEC_copy(VEC *restrict self, const VEC *restrict va) {
    VV_FIT(self, va);

    ITER_VEC_by_16(self, i) {
        LOAD_VEC_by_16(y, va, i);
        STORE_VEC_by_16(self, i, y);
    }
}

void VEC_resize(VEC *self, size_t dim) {
    assert(dim < self->dim);

    memset(self->data + dim, 0, (self->padded_dim - dim) * sizeof(double));

    self->dim        = dim;
    self->padded_dim = PAD(dim, N_VEC_PAD);
}

void VEC_shift(VEC *self, size_t r, size_t shift) {
    assert(r < self->dim);
    assert(shift < self->dim);

    if (shift == 0) {
        double val = VVAL(self, r);

        ITER_VEC(self, i) {
            if (i < r)
                VVAL(self, i) -= val;
            else if (i > r)
                VVAL(self, i - 1) = VVAL(self, i) - val;
        }
        VEC_resize(self, self->dim - 1);
    }
    else {
        size_t t      = (r + shift) % self->dim;
        VVAL(self, 0) = VVAL(self, t) - VVAL(self, r);
        VEC_resize(self, 1);
    }
}

void VEC_relu(VEC *self) {
    const __m256d zero = _mm256_setzero_pd();
    ITER_VEC_by_16(self, i) {
        LOAD_VEC_by_16(y, self, i);
        for (size_t k = 0; k < 4; ++k)
            y[k] = _mm256_max_pd(y[k], zero);
        STORE_VEC_by_16(self, i, y);
    }
}

void VEC_set(VEC *self, double val) {
    LOAD_CONST_by_16(y, val);
    ITER_VEC_by_16(self, i) {
        STORE_VEC_by_16(self, i, y);
    }
}

void VEC_clear(VEC *self) {
    VEC_set(self, 0.);
}

void VEC_div(VEC *self, size_t n) {
    assert(n != 0);

    const __m256d ymul = _mm256_set1_pd(1. / n);
    ITER_VEC_by_16(self, i) {
        LOAD_VEC_by_16(y, self, i);
        for (size_t k = 0; k < 4; ++k)
            y[k] = _mm256_mul_pd(y[k], ymul);
        STORE_VEC_by_16(self, i, y);
    }
}

size_t VEC_argmax(const VEC *self) {
    double max = -DBL_MAX;
    size_t ret = 0;

    ITER_VEC(self, i) {
        if (VVAL(self, i) > max) {
            max = VVAL(self, i);
            ret = i;
        }
    }

    return ret;
}

bool VEC_any_pos(const VEC *self) {
    ITER_VEC(self, i) {
        if (VVAL(self, i) > EPS)
            return true;
    }
    return false;
}

// ---------------------------------- Matrix -----------------------------------
#define LOAD_MAT(self, r, offset) \
    _mm256_load_pd(self->data[r] + (offset))

#define LOAD_MAT_by_16(var, self, r, offset)  \
    __m256d var[4];                           \
    var[0] = LOAD_MAT(self, r, (offset));     \
    var[1] = LOAD_MAT(self, r, (offset) + 4); \
    var[2] = LOAD_MAT(self, r, (offset) + 8); \
    var[3] = LOAD_MAT(self, r, (offset) + 12)

#define STORE_MAT(self, r, offset, var) \
    _mm256_store_pd(self->data[r] + offset, var)

#define STORE_MAT_by_16(self, r, offset, var)  \
    STORE_MAT(self, r, (offset),      var[0]); \
    STORE_MAT(self, r, (offset) + 4,  var[1]); \
    STORE_MAT(self, r, (offset) + 8,  var[2]); \
    STORE_MAT(self, r, (offset) + 12, var[3])

#define MV_FIT(m, v) \
    assert(m->dim2 == v->dim)

#define VM_FIT(v, m) \
    assert(v->dim == m->dim1)

MAT *MAT_alloc(size_t dim1, size_t dim2) {
    MAT *self = XMALLOC(sizeof(MAT));

    self->dim1 = dim1;
    self->dim2 = dim2;
    self->data = XMALLOC(self->dim1 * sizeof(double *));
    self->rows = XMALLOC(self->dim1 * sizeof(VEC *));

    ITER_MAT1(self, i) {
        self->rows[i] = VEC_alloc(dim2);
        self->data[i] = self->rows[i]->data;
    }

    return self;
}

MAT *MAT_alloc_fp(size_t dim1, size_t dim2, FILE *fp) {
    MAT *self = MAT_alloc(dim1, dim2);

    ITER_MAT1(self, i) {
        ITER_MAT2(self, j) {
            fscanf(fp, "%lf", &MVAL(self, i, j));
        }
    }

    return self;
}

MAT *MAT_alloc_file(size_t dim1, size_t dim2, const char *mat_path) {
    FILE *fp   = XFOPEN(mat_path, "r");
    MAT  *self = MAT_alloc_fp(dim1, dim2, fp);
    fclose(fp);

    return self;
}

void MAT_free(MAT *self) {
    ITER_MAT1(self, i) {
        VEC_free(self->rows[i]);
    }
    free(self->data);
    free(self->rows);
    free(self);
}

void MAT_dump(MAT *self) {
    ITER_MAT1(self, i) {
        ITER_MAT2(self, j) {
            printf("%6.2lf ", MVAL(self, i, j));
        }
        printf("\n");
    }
}

void MAT_resize(MAT *self, size_t dim1) {
    assert(dim1 <= self->dim1);

    for (size_t i = dim1; i < self->dim1; ++i) {
        VEC_free(self->rows[i]);
        self->rows[i] = NULL;
        self->data[i] = NULL;
    }
    self->dim1 = dim1;
}

void MAT_shift(MAT *self, size_t r, size_t shift) {
    assert(r < self->dim1);
    assert(shift < self->dim1);

    if (shift == 0) {
        VEC *self_r = VEC_alloc(self->dim2);
        VEC_copy(self_r, MSLICE(self, r));

        ITER_MAT1(self, i) {
            if (i < r)
                VV_sub(MSLICE(self, i), MSLICE(self, i), self_r);
            else if (i > r)
                VV_sub(MSLICE(self, i - 1), MSLICE(self, i), self_r);
        }
        MAT_resize(self, self->dim1 - 1);

        VEC_free(self_r);
    }
    else {
        size_t t = (r + shift) % self->dim1;
        VV_sub(MSLICE(self, 0), MSLICE(self, t), MSLICE(self, r));

        MAT_resize(self, 1);
    }
}

// ------------------------------- Vector-Vector -------------------------------
// binary ops, e.g. add/sub/max/min..., the third argument is for zero.
typedef __m256d (*avx_binop_pd_t)(__m256d, __m256d, __m256d);

static inline __m256d op_add(__m256d a, __m256d b,
                             __m256d zero __attribute__((unused))) {
    return _mm256_add_pd(a, b);
}

static inline __m256d op_sub(__m256d a, __m256d b,
                             __m256d zero __attribute__((unused))) {
    return _mm256_sub_pd(a, b);
}

static inline __m256d op_max(__m256d a, __m256d b,
                             __m256d zero __attribute__((unused))) {
    return _mm256_max_pd(a, b);
}

static inline __m256d op_min(__m256d a, __m256d b,
                             __m256d zero __attribute__((unused))) {
    return _mm256_min_pd(a, b);
}

// a + max(b, 0);
static inline __m256d op_add_pos(__m256d a, __m256d b, __m256d zero) {
    return _mm256_add_pd(a, _mm256_max_pd(b, zero));
}

// a + min(b, 0);
static inline __m256d op_add_neg(__m256d a, __m256d b, __m256d zero) {
    return _mm256_add_pd(a, _mm256_min_pd(b, zero));
}

// a - max(b, 0);
static inline __m256d op_sub_pos(__m256d a, __m256d b, __m256d zero) {
    return _mm256_sub_pd(a, _mm256_max_pd(b, zero));
}

// a - min(b, 0);
static inline __m256d op_sub_neg(__m256d a, __m256d b, __m256d zero) {
    return _mm256_sub_pd(a, _mm256_min_pd(b, zero));
}

// vdest <- op(va, vb)
static void VV_binop(VEC *vdest, const VEC *va, const VEC *vb,
                     avx_binop_pd_t op) {
    VV_FIT(vdest, va);
    VV_FIT(vdest, vb);

    const __m256d zero = _mm256_setzero_pd();
    ITER_VEC_by_16(va, i) {
        LOAD_VEC_by_16(ya, va, i);
        LOAD_VEC_by_16(yb, vb, i);
        for (size_t k = 0; k < 4; ++k)
            ya[k] = op(ya[k], yb[k], zero);
        STORE_VEC_by_16(vdest, i, ya);
    }
}

void VV_add(VEC *vdest, const VEC *va, const VEC *vb) {
    VV_binop(vdest, va, vb, op_add);
}

void VV_sub(VEC *vdest, const VEC *va, const VEC *vb) {
    VV_binop(vdest, va, vb, op_sub);
}

void VV_sub_pos(VEC *vdest, const VEC *va) {
    VV_binop(vdest, vdest, va, op_sub_pos);
}

void VV_sub_neg(VEC *vdest, const VEC *va) {
    VV_binop(vdest, vdest, va, op_sub_neg);
}

// ------------------------------- Matrix-Vector -------------------------------
// vdest <- op(op(va, mb[index[0]]), m[index[1]]) ...
static void MV_binop(VEC *vdest, const VEC *va, const MAT *restrict mb,
                     BArray *ba, avx_binop_pd_t op) {
    MV_FIT(mb, vdest);
    MV_FIT(mb, va);

    bool      all_rows = (ba == NULL);
    uint16_t *index    = all_rows ? NULL : BArray_list(ba);
    size_t    size     = all_rows ? mb->dim1 : BArray_counter(ba);

    const __m256d zero = _mm256_setzero_pd();

    ITER_VEC_by_16(vdest, i) {
        LOAD_VEC_by_16(ya, va, i);

        size_t j = 0;
        for (; j + 7 < size; j += 8) {
            LOAD_MAT_by_16(yb0, mb, (all_rows ? j : index[j]), i);
            LOAD_MAT_by_16(yb1, mb, (all_rows ? (j + 1) : index[j + 1]), i);
            LOAD_MAT_by_16(yb2, mb, (all_rows ? (j + 2) : index[j + 2]), i);
            LOAD_MAT_by_16(yb3, mb, (all_rows ? (j + 3) : index[j + 3]), i);
            LOAD_MAT_by_16(yb4, mb, (all_rows ? (j + 4) : index[j + 4]), i);
            LOAD_MAT_by_16(yb5, mb, (all_rows ? (j + 5) : index[j + 5]), i);
            LOAD_MAT_by_16(yb6, mb, (all_rows ? (j + 6) : index[j + 6]), i);
            LOAD_MAT_by_16(yb7, mb, (all_rows ? (j + 7) : index[j + 7]), i);

            for (size_t k = 0; k < 4; ++k) {
                ya[k] = op(ya[k], yb0[k], zero);
                ya[k] = op(ya[k], yb1[k], zero);
                ya[k] = op(ya[k], yb2[k], zero);
                ya[k] = op(ya[k], yb3[k], zero);
                ya[k] = op(ya[k], yb4[k], zero);
                ya[k] = op(ya[k], yb5[k], zero);
                ya[k] = op(ya[k], yb6[k], zero);
                ya[k] = op(ya[k], yb7[k], zero);
            }
        }
        for (; j < size; ++j) {
            LOAD_MAT_by_16(yb, mb, (all_rows ? j : index[j]), i);

            for (size_t k = 0; k < 4; ++k)
                ya[k] = op(ya[k], yb[k], zero);
        }

        STORE_VEC_by_16(vdest, i, ya);
    }
}

void MV_acc(VEC *vdest, const VEC *va, const MAT *restrict mb, BArray *ba) {
    if (va == NULL) {
        VEC_clear(vdest);
        MV_binop(vdest, vdest, mb, ba, op_add);
    }
    else {
        MV_binop(vdest, va, mb, ba, op_add);
    }
}

void MV_acc_pos(VEC *vdest, const VEC *va, const MAT *restrict mb, BArray *ba) {
    if (va == NULL) {
        VEC_clear(vdest);
        MV_binop(vdest, vdest, mb, ba, op_add_pos);
    }
    else {
        MV_binop(vdest, va, mb, ba, op_add_pos);
    }
}

void MV_acc_neg(VEC *vdest, const VEC *va, const MAT *restrict mb, BArray *ba) {
    if (va == NULL) {
        VEC_clear(vdest);
        MV_binop(vdest, vdest, mb, ba, op_add_neg);
    }
    else {
        MV_binop(vdest, va, mb, ba, op_add_neg);
    }
}

void MV_max(VEC *vdest, const VEC *va, const MAT *restrict mb, BArray *ba) {
    if (va == NULL) {
        if (BArray_counter(ba) == 0) {
            VEC_clear(vdest);
        }
        else {
            VEC_set(vdest, -DBL_MAX);
            MV_binop(vdest, vdest, mb, ba, op_max);
        }
    }
    else {
        MV_binop(vdest, va, mb, ba, op_max);
    }
}

void MV_min(VEC *vdest, const VEC *va, const MAT *restrict mb, BArray *ba) {
    if (va == NULL) {
        if (BArray_counter(ba) == 0) {
            VEC_clear(vdest);
        }
        else {
            VEC_set(vdest, DBL_MAX);
            MV_binop(vdest, vdest, mb, ba, op_min);
        }
    }
    else {
        MV_binop(vdest, va, mb, ba, op_min);
    }
}

void MV_dot(VEC *restrict vdest, const MAT *restrict ma,
            const VEC *restrict vb) {
    VM_FIT(vdest, ma);
    MV_FIT(ma, vb);

    size_t i = 0;
    for (; i + 3 < vdest->dim; i += 4) {
        LOAD_CONST_by_16(acc, 0.);

        ITER_VEC_by_16(vb, j) {
            LOAD_MAT_by_16(ya0, ma, i, j);
            LOAD_MAT_by_16(ya1, ma, i + 1, j);
            LOAD_MAT_by_16(ya2, ma, i + 2, j);
            LOAD_MAT_by_16(ya3, ma, i + 3, j);
            LOAD_VEC_by_16(yb, vb, j);

            for (size_t k = 0; k < 4; ++k) {
                acc[0] = _mm256_fmadd_pd(ya0[k], yb[k], acc[0]);
                acc[1] = _mm256_fmadd_pd(ya1[k], yb[k], acc[1]);
                acc[2] = _mm256_fmadd_pd(ya2[k], yb[k], acc[2]);
                acc[3] = _mm256_fmadd_pd(ya3[k], yb[k], acc[3]);
            }
        }

        STORE_VEC(vdest, i, hsum(acc));
    }
    for (; i < vdest->dim; ++i) {
        __m256d acc = _mm256_setzero_pd();

        ITER_VEC_by_16(vb, j) {
            LOAD_MAT_by_16(ya, ma, i, j);
            LOAD_VEC_by_16(yb, vb, j);

            for (size_t k = 0; k < 4; ++k)
                acc = _mm256_fmadd_pd(ya[k], yb[k], acc);
        }

        double tmp[4];
        _mm256_store_pd(tmp, acc);
        VVAL(vdest, i) = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    }
}

void MV_relax(VEC *restrict vdestl, VEC *restrict vdestu,
              const MAT *restrict ma, const MAT *restrict ma_abs,
              const VEC *restrict vbl, const VEC *restrict vbu,
              VEC *restrict tmps, VEC *restrict tmpd) {
    MV_FIT(ma, vbl);
    MV_FIT(ma, vbu);
    VM_FIT(vdestl, ma);
    VM_FIT(vdestu, ma);

    __m256d half = _mm256_set1_pd(.5);
    ITER_VEC_by_16(vbl, i) {
        LOAD_VEC_by_16(ybl, vbl, i);
        LOAD_VEC_by_16(ybu, vbu, i);
        __m256d ys[4];
        __m256d yd[4];
        for (size_t k = 0; k < 4; ++k) {
            ys[k] = _mm256_mul_pd(_mm256_add_pd(ybu[k], ybl[k]), half);
            yd[k] = _mm256_mul_pd(_mm256_sub_pd(ybu[k], ybl[k]), half);
        }
        STORE_VEC_by_16(tmps, i, ys);
        STORE_VEC_by_16(tmpd, i, yd);
    }

    size_t i = 0;
    for (; i + 3 < vdestl->dim; i += 4) {
        LOAD_CONST_by_16(accs, 0.);
        LOAD_CONST_by_16(accd, 0.);

        ITER_VEC_by_16(vbl, j) {
            LOAD_MAT_by_16(yas0, ma, i, j);
            LOAD_MAT_by_16(yas1, ma, i + 1, j);
            LOAD_MAT_by_16(yas2, ma, i + 2, j);
            LOAD_MAT_by_16(yas3, ma, i + 3, j);
            LOAD_MAT_by_16(yad0, ma_abs, i, j);
            LOAD_MAT_by_16(yad1, ma_abs, i + 1, j);
            LOAD_MAT_by_16(yad2, ma_abs, i + 2, j);
            LOAD_MAT_by_16(yad3, ma_abs, i + 3, j);
            LOAD_VEC_by_16(ys, tmps, j);
            LOAD_VEC_by_16(yd, tmpd, j);

            for (size_t k = 0; k < 4; ++k) {
                accs[0] = _mm256_fmadd_pd(yas0[k], ys[k], accs[0]);
                accs[1] = _mm256_fmadd_pd(yas1[k], ys[k], accs[1]);
                accs[2] = _mm256_fmadd_pd(yas2[k], ys[k], accs[2]);
                accs[3] = _mm256_fmadd_pd(yas3[k], ys[k], accs[3]);

                accd[0] = _mm256_fmadd_pd(yad0[k], yd[k], accd[0]);
                accd[1] = _mm256_fmadd_pd(yad1[k], yd[k], accd[1]);
                accd[2] = _mm256_fmadd_pd(yad2[k], yd[k], accd[2]);
                accd[3] = _mm256_fmadd_pd(yad3[k], yd[k], accd[3]);
            }
        }

        __m256d s  = hsum(accs);
        __m256d d  = hsum(accd);
        __m256d lb = _mm256_sub_pd(s, d);
        __m256d ub = _mm256_add_pd(s, d);
        STORE_VEC(vdestl, i, lb);
        STORE_VEC(vdestu, i, ub);
    }
    for (; i < vdestl->dim; ++i) {
        __m256d accs = _mm256_setzero_pd();
        __m256d accd = _mm256_setzero_pd();

        ITER_VEC_by_16(vbl, j) {
            LOAD_MAT_by_16(yas, ma, i, j);
            LOAD_MAT_by_16(yad, ma_abs, i, j);
            LOAD_VEC_by_16(ys, tmps, j);
            LOAD_VEC_by_16(yd, tmpd, j);

            for (size_t k = 0; k < 4; ++k) {
                accs = _mm256_fmadd_pd(yas[k], ys[k], accs);
                accd = _mm256_fmadd_pd(yad[k], yd[k], accd);
            }
        }

        double tmps[4], tmpd[4];
        _mm256_store_pd(tmps, accs);
        _mm256_store_pd(tmpd, accd);

        double s        = tmps[0] + tmps[1] + tmps[2] + tmps[3];
        double d        = tmpd[0] + tmpd[1] + tmpd[2] + tmpd[3];
        VVAL(vdestl, i) = s - d;
        VVAL(vdestu, i) = s + d;
    }
}

// ------------------------------- Matrix-Matrix -------------------------------
void MM_dot(MAT *restrict mdest, const MAT *restrict ma,
            const MAT *restrict mb) {
    assert(mdest->dim1 == ma->dim1);
    assert(mdest->dim2 == mb->dim2);
    assert(ma->dim2 == mb->dim1);

    ITER_MAT1(mdest, i) {
        ITER_MAT2(mdest, j) {
            MVAL(mdest, i, j) = 0.;
            ITER_MAT2(ma, k) {
                MVAL(mdest, i, j) += MVAL(ma, i, k) * MVAL(mb, k, j);
            }
        }
    }
}

void MM_dot_trans(MAT *restrict mdest, const MAT *restrict ma,
                  const MAT *restrict mb) {
    assert(mdest->dim1 == ma->dim1);
    assert(mdest->dim2 == mb->dim1);
    assert(ma->dim2 == mb->dim2);

    ITER_MAT1(mdest, i) {
        MV_dot(MSLICE(mdest, i), mb, MSLICE(ma, i));
    }
}

// ---------------------------------- sorted -----------------------------------
void MV_trinity(VEC *restrict vacc, VEC *restrict vmax, VEC *restrict vmin,
                const MAT *restrict ma, BArray *ba) {
    MV_FIT(ma, vacc);
    MV_FIT(ma, vmax);
    MV_FIT(ma, vmin);

    uint16_t *index = BArray_list(ba);
    size_t    size  = BArray_counter(ba);

    ITER_VEC_by_16(vacc, i) {
        LOAD_VEC_by_16(yacc, vacc, i);
        LOAD_CONST_by_16(ymax, -DBL_MAX);
        LOAD_CONST_by_16(ymin, DBL_MAX);

        size_t j = 0;
        for (; j + 7 < size; j += 8) {
            LOAD_MAT_by_16(ya0, ma, index[j], i);
            LOAD_MAT_by_16(ya1, ma, index[j + 1], i);
            LOAD_MAT_by_16(ya2, ma, index[j + 2], i);
            LOAD_MAT_by_16(ya3, ma, index[j + 3], i);
            LOAD_MAT_by_16(ya4, ma, index[j + 4], i);
            LOAD_MAT_by_16(ya5, ma, index[j + 5], i);
            LOAD_MAT_by_16(ya6, ma, index[j + 6], i);
            LOAD_MAT_by_16(ya7, ma, index[j + 7], i);

            for (size_t k = 0; k < 4; ++k) {
                yacc[k] = _mm256_add_pd(yacc[k], ya0[k]);
                yacc[k] = _mm256_add_pd(yacc[k], ya1[k]);
                yacc[k] = _mm256_add_pd(yacc[k], ya2[k]);
                yacc[k] = _mm256_add_pd(yacc[k], ya3[k]);
                yacc[k] = _mm256_add_pd(yacc[k], ya4[k]);
                yacc[k] = _mm256_add_pd(yacc[k], ya5[k]);
                yacc[k] = _mm256_add_pd(yacc[k], ya6[k]);
                yacc[k] = _mm256_add_pd(yacc[k], ya7[k]);

                ymax[k] = _mm256_max_pd(ymax[k], ya0[k]);
                ymax[k] = _mm256_max_pd(ymax[k], ya1[k]);
                ymax[k] = _mm256_max_pd(ymax[k], ya2[k]);
                ymax[k] = _mm256_max_pd(ymax[k], ya3[k]);
                ymax[k] = _mm256_max_pd(ymax[k], ya4[k]);
                ymax[k] = _mm256_max_pd(ymax[k], ya5[k]);
                ymax[k] = _mm256_max_pd(ymax[k], ya6[k]);
                ymax[k] = _mm256_max_pd(ymax[k], ya7[k]);

                ymin[k] = _mm256_min_pd(ymin[k], ya0[k]);
                ymin[k] = _mm256_min_pd(ymin[k], ya1[k]);
                ymin[k] = _mm256_min_pd(ymin[k], ya2[k]);
                ymin[k] = _mm256_min_pd(ymin[k], ya3[k]);
                ymin[k] = _mm256_min_pd(ymin[k], ya4[k]);
                ymin[k] = _mm256_min_pd(ymin[k], ya5[k]);
                ymin[k] = _mm256_min_pd(ymin[k], ya6[k]);
                ymin[k] = _mm256_min_pd(ymin[k], ya7[k]);
            }
        }
        for (; j < size; ++j) {
            LOAD_MAT_by_16(ya, ma, index[j], i);

            for (size_t k = 0; k < 4; ++k) {
                yacc[k] = _mm256_add_pd(yacc[k], ya[k]);

                ymax[k] = _mm256_max_pd(ymax[k], ya[k]);

                ymin[k] = _mm256_min_pd(ymin[k], ya[k]);
            }
        }

        STORE_VEC_by_16(vacc, i, yacc);
        STORE_VEC_by_16(vmax, i, ymax);
        STORE_VEC_by_16(vmin, i, ymin);
    }
}

static void VV_top_2_of_3(VEC *restrict vdest1, VEC *restrict vdest2,
                          VEC *restrict va) {
    VV_FIT(vdest1, vdest2);
    VV_FIT(vdest1, va);

    __m256d tmp;
    ITER_VEC_by_16(va, i) {
        LOAD_VEC_by_16(ydest1, vdest1, i);
        LOAD_VEC_by_16(ydest2, vdest2, i);
        LOAD_VEC_by_16(ya, va, i);

        for (size_t k = 0; k < 4; ++k) {
            CSWAP(ydest1[k], ya[k]);
            CSWAP(ydest2[k], ya[k]);
        }

        STORE_VEC_by_16(vdest1, i, ydest1);
        STORE_VEC_by_16(vdest2, i, ydest2);
    }
}

static void VV_bot_2_of_3(VEC *restrict vdest1, VEC *restrict vdest2,
                          VEC *restrict va) {
    VV_FIT(vdest1, vdest2);
    VV_FIT(vdest1, va);

    __m256d tmp;
    ITER_VEC_by_16(va, i) {
        LOAD_VEC_by_16(ydest1, vdest1, i);
        LOAD_VEC_by_16(ydest2, vdest2, i);
        LOAD_VEC_by_16(ya, va, i);

        for (size_t k = 0; k < 4; ++k) {
            CSWAP(ya[k], ydest1[k]);
            CSWAP(ya[k], ydest2[k]);
        }

        STORE_VEC_by_16(vdest1, i, ydest1);
        STORE_VEC_by_16(vdest2, i, ydest2);
    }
}

static void VV_top_2_of_4(VEC *restrict vdest1, VEC *restrict vdest2,
                          VEC *restrict va, VEC *restrict vb) {
    VV_FIT(vdest1, vdest2);
    VV_FIT(vdest1, va);
    VV_FIT(vdest1, vb);

    __m256d tmp;
    ITER_VEC_by_16(va, i) {
        LOAD_VEC_by_16(ydest1, vdest1, i);
        LOAD_VEC_by_16(ydest2, vdest2, i);
        LOAD_VEC_by_16(ya, va, i);
        LOAD_VEC_by_16(yb, vb, i);

        for (size_t k = 0; k < 4; ++k) {
            CSWAP(ydest1[k], ya[k]);
            CSWAP(ydest1[k], yb[k]);
            CSWAP(ydest2[k], ya[k]);
            CSWAP(ydest2[k], yb[k]);
        }

        STORE_VEC_by_16(vdest1, i, ydest1);
        STORE_VEC_by_16(vdest2, i, ydest2);
    }
}

static void VV_bot_2_of_4(VEC *restrict vdest1, VEC *restrict vdest2,
                          VEC *restrict va, VEC *restrict vb) {
    VV_FIT(vdest1, vdest2);
    VV_FIT(vdest1, va);
    VV_FIT(vdest1, vb);

    __m256d tmp;
    ITER_VEC_by_16(va, i) {
        LOAD_VEC_by_16(ydest1, vdest1, i);
        LOAD_VEC_by_16(ydest2, vdest2, i);
        LOAD_VEC_by_16(ya, va, i);
        LOAD_VEC_by_16(yb, vb, i);

        for (size_t k = 0; k < 4; ++k) {
            CSWAP(ya[k], ydest1[k]);
            CSWAP(yb[k], ydest1[k]);
            CSWAP(ya[k], ydest2[k]);
            CSWAP(yb[k], ydest2[k]);
        }

        STORE_VEC_by_16(vdest1, i, ydest1);
        STORE_VEC_by_16(vdest2, i, ydest2);
    }
}

void MV_max2(VEC *restrict vdest1, VEC *restrict vdest2, const MAT *restrict ma,
             BArray *ba) {
    MV_FIT(ma, vdest1);
    MV_FIT(ma, vdest2);

    uint16_t *index = BArray_list(ba);
    size_t    size  = BArray_counter(ba);

    VEC_set(vdest1, -DBL_MAX);
    VEC_set(vdest2, -DBL_MAX);

    size_t i = 0;
    for (; i + 1 < size; i += 2)
        VV_top_2_of_4(vdest1, vdest2, MSLICE(ma, index[i]),
                      MSLICE(ma, index[i + 1]));
    for (; i < size; ++i)
        VV_top_2_of_3(vdest1, vdest2, MSLICE(ma, index[i]));
}

void MV_min2(VEC *restrict vdest1, VEC *restrict vdest2, const MAT *restrict ma,
             BArray *ba) {
    MV_FIT(ma, vdest1);
    MV_FIT(ma, vdest2);

    uint16_t *index = BArray_list(ba);
    size_t    size  = BArray_counter(ba);

    VEC_set(vdest1, DBL_MAX);
    VEC_set(vdest2, DBL_MAX);

    size_t i = 0;
    for (; i + 1 < size; i += 2)
        VV_bot_2_of_4(vdest1, vdest2, MSLICE(ma, index[i]),
                      MSLICE(ma, index[i + 1]));
    for (; i < size; ++i)
        VV_bot_2_of_3(vdest1, vdest2, MSLICE(ma, index[i]));
}
