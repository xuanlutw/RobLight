#include <assert.h>
#include <float.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <emmintrin.h>
#include <immintrin.h>

#include "Matrix.h"
#include "utils.h"

// -----------------------------------------------------------------------------
static inline uint16_t lookup(const uint16_t *list, uint16_t k) {
    return list ? list[k] : k;
}

#define LOAD_by_8(var, ptr)           \
    var[0] = _mm256_load_pd(ptr);     \
    var[1] = _mm256_load_pd(ptr + 4); \
    ptr += 8

#define STORE_by_8(ptr, var)          \
    _mm256_store_pd(ptr, var[0]);     \
    _mm256_store_pd(ptr + 4, var[1]); \
    ptr += 8

static inline double hsum1(__m256d y) {
    __m128d yl = _mm256_castpd256_pd128(y);
    __m128d yh = _mm256_extractf128_pd(y, 1);

    __m128d ysum = _mm_add_pd(yl, yh);

    __m128d yshf = _mm_unpackhi_pd(ysum, ysum);

    return _mm_cvtsd_f64(_mm_add_sd(ysum, yshf));
}

static inline __m128d hsum2(__m256d y0, __m256d y1) {
    __m128d yl0 = _mm256_castpd256_pd128(y0);
    __m128d yh0 = _mm256_extractf128_pd(y0, 1);
    __m128d yl1 = _mm256_castpd256_pd128(y1);
    __m128d yh1 = _mm256_extractf128_pd(y1, 1);

    __m128d ysum0 = _mm_add_pd(yl0, yh0);
    __m128d ysum1 = _mm_add_pd(yl1, yh1);

    return _mm_hadd_pd(ysum0, ysum1);
}

static inline __m256d hsum4(__m256d y[4]) {
    __m256d t0 = _mm256_hadd_pd(y[0], y[1]);
    __m256d t1 = _mm256_hadd_pd(y[2], y[3]);

    __m256d u0 = _mm256_permute2f128_pd(t0, t1, 0x20);
    __m256d u1 = _mm256_permute2f128_pd(t0, t1, 0x31);

    return _mm256_add_pd(u0, u1);
}

#define CSWAP(a, b)            \
    tmp = _mm256_max_pd(a, b); \
    b   = _mm256_min_pd(a, b); \
    a   = tmp

#define VV_FIT(v1, v2) \
    (VEC_dim(v1) == VEC_dim(v2))

#define M1V_FIT(m, v) \
    (MAT_dim1(m) == VEC_dim(v))

#define M2V_FIT(m, v) \
    (MAT_dim2(m) == VEC_dim(v))

#define M1M1_FIT(m1, m2) \
    (MAT_dim1(m1) == MAT_dim1(m2))

#define M1M2_FIT(m1, m2) \
    (MAT_dim1(m1) == MAT_dim2(m2))

#define M2M1_FIT(m1, m2) \
    (MAT_dim2(m1) == MAT_dim1(m2))

#define M2M2_FIT(m1, m2) \
    (MAT_dim2(m1) == MAT_dim2(m2))

// ---------------------------------- Vector -----------------------------------
VEC VEC_alloc(size_t dim) {
    size_t      pad_dim    = PAD(dim, N_VEC_PAD);
    size_t      chunk_size = PAD_VEC_HEADER_SIZE + pad_dim * sizeof(double);
    VEC_header *header     = XMALLOC(chunk_size);

    header->dim     = dim;
    header->pad_dim = pad_dim;

    VEC self = (VEC)((char *)header + PAD_VEC_HEADER_SIZE);
    memset(self, 0, pad_dim * sizeof(double));

    return self;
}

VEC VEC_alloc_fp(size_t dim, FILE *fp) {
    VEC self = VEC_alloc(dim);

    ITER_VEC(self, i) {
        fscanf(fp, "%lf", &(self[i]));
    }

    return self;
}

void VEC_free(VEC self) {
    VEC_header *header = VEC_get_header(self);
    free(header);
}

void VEC_dump(VEC self) {
    ITER_VEC(self, i) {
        printf("%6.2lf ", self[i]);
    }
    printf("\n");
}

void VEC_resize(VEC self, size_t dim) {
    assert(dim < VEC_dim(self));

    VEC_header *header = VEC_get_header(self);
    memset(self + dim, 0, (header->pad_dim - dim) * sizeof(double));

    header->dim     = dim;
    header->pad_dim = PAD(dim, N_VEC_PAD);
}

void VEC_shift(VEC self, size_t r, size_t shift) {
    assert(r < VEC_dim(self));
    assert(shift < VEC_dim(self));

    if (shift == 0) {
        double val = self[r];

        ITER_VEC(self, i) {
            if (i < r)
                self[i] -= val;
            else if (i > r)
                self[i - 1] = self[i] - val;
        }
        VEC_resize(self, VEC_dim(self) - 1);
    }
    else {
        size_t t = (r + shift) % VEC_dim(self);
        self[0]  = self[t] - self[r];
        VEC_resize(self, 1);
    }
}

void VEC_copy(VEC_restr self, VEC_restr va) {
    assert(VV_FIT(self, va));

    memcpy(self, va, VEC_dim(self) * sizeof(double));
}

void VEC_clear(VEC self) {
    memset(self, 0, VEC_pad_dim(self) * sizeof(double));
}

void VEC_set(VEC self, double val) {
    const size_t  dim  = VEC_dim(self);
    const __m256d yval = _mm256_set1_pd(val);

    double *st_self = self;

    size_t i = 0;
    for (; i + 7 < dim; i += 8) {  // no over
        _mm256_store_pd(st_self, yval);
        _mm256_store_pd(st_self + 4, yval);
        st_self += 8;
    }
    for (; i < dim; ++i)
        self[i] = val;
}

void VEC_div(VEC self, size_t n) {
    assert(n != 0);

    const size_t  pad_dim = VEC_pad_dim(self);
    const __m256d ymul    = _mm256_set1_pd(1. / n);

    double *ld_self = self;
    double *st_self = self;

    for (size_t i = 0; i < pad_dim; i += 8) {
        __m256d y[2];
        LOAD_by_8(y, ld_self);

        y[0] = _mm256_mul_pd(y[0], ymul);
        y[1] = _mm256_mul_pd(y[1], ymul);

        STORE_by_8(st_self, y);
    }
}

void VEC_relu(VEC self) {
    const size_t  pad_dim = VEC_pad_dim(self);
    const __m256d zero    = _mm256_setzero_pd();

    double *ld_self = self;
    double *st_self = self;

    for (size_t i = 0; i < pad_dim; i += 8) {
        __m256d y[2];
        LOAD_by_8(y, ld_self);

        y[0] = _mm256_max_pd(y[0], zero);
        y[1] = _mm256_max_pd(y[1], zero);

        STORE_by_8(st_self, y);
    }
}

size_t VEC_argmax(VEC self) {
    double max = -DBL_MAX;
    size_t ret = 0;

    ITER_VEC(self, i) {
        if (self[i] > max) {
            max = self[i];
            ret = i;
        }
    }

    return ret;
}

bool VEC_any_pos(VEC self) {
    ITER_VEC(self, i) {
        if (self[i] > EPS)
            return true;
    }
    return false;
}

// ---------------------------------- Matrix -----------------------------------
MAT MAT_alloc(size_t dim1, size_t dim2) {
    size_t      chunk_size = PAD_MAT_HEADER_SIZE + dim1 * sizeof(VEC);
    MAT_header *header     = XMALLOC(chunk_size);

    header->dim1     = dim1;
    header->dim2     = dim2;
    header->pad_dim2 = PAD(dim2, N_VEC_PAD);

    header->abs = NULL;

    MAT self = (MAT)((char *)header + PAD_MAT_HEADER_SIZE);
    ITER_MAT1(self, i) {
        self[i] = VEC_alloc(dim2);
    }

    return self;
}

MAT MAT_alloc_fp(size_t dim1, size_t dim2, FILE *fp) {
    MAT self = MAT_alloc(dim1, dim2);

    ITER_MAT1(self, i) {
        ITER_MAT2(self, j) {
            fscanf(fp, "%lf", &(self[i][j]));
        }
    }

    return self;
}

MAT MAT_alloc_file(size_t dim1, size_t dim2, const char *mat_path) {
    FILE *fp   = XFOPEN(mat_path, "r");
    MAT   self = MAT_alloc_fp(dim1, dim2, fp);
    fclose(fp);

    return self;
}

void MAT_free(MAT self) {
    MAT abs = MAT_abs(self);
    if (abs != NULL) {
        MAT_free(abs);
    }

    ITER_MAT1(self, i) {
        VEC_free(self[i]);
    }

    MAT_header *header = MAT_get_header(self);
    free(header);
}

void MAT_dump(MAT self) {
    ITER_MAT1(self, i) {
        ITER_MAT2(self, j) {
            printf("%6.2lf ", self[i][j]);
        }
        printf("\n");
    }

    MAT abs = MAT_abs(self);
    if (abs != NULL) {
        printf("\n");
        MAT_dump(abs);
    }
}

void MAT_comp_abs(MAT self) {
    MAT_header *header = MAT_get_header(self);
    size_t      dim1   = header->dim1;
    size_t      dim2   = header->dim2;

    MAT abs = MAT_alloc(dim1, dim2);
    ITER_MAT1(self, i) {
        ITER_MAT2(self, j) {
            abs[i][j] = ABS(self[i][j]);
        }
    }

    header->abs = abs;
}

void MAT_resize(MAT self, size_t dim1) {
    assert(dim1 <= MAT_dim1(self));

    MAT_header *header = MAT_get_header(self);

    for (size_t i = dim1; i < header->dim1; ++i)
        VEC_free(self[i]);
    header->dim1 = dim1;
}

void MAT_shift(MAT self, size_t r, size_t shift) {
    assert(r < MAT_dim1(self));
    assert(shift < MAT_dim1(self));

    if (shift == 0) {
        VEC self_r = VEC_alloc(MAT_dim2(self));
        VEC_copy(self_r, self[r]);

        ITER_MAT1(self, i) {
            if (i < r)
                VV_sub(self[i], self[i], self_r);
            else if (i > r)
                VV_sub(self[i - 1], self[i], self_r);
        }
        MAT_resize(self, MAT_dim1(self) - 1);

        VEC_free(self_r);
    }
    else {
        size_t t = (r + shift) % MAT_dim1(self);
        VV_sub(self[0], self[t], self[r]);
        MAT_resize(self, 1);
    }
}

// ------------------------------- Vector-Vector -------------------------------
// binary ops, e.g. add/sub/max/min..., the third argument is for zero.
typedef __m256d (*binop_pd_t)(__m256d, __m256d, __m256d);

static inline __m256d op_add(__m256d a, __m256d b,
                             __m256d zero __attribute__((unused))) {
    return _mm256_add_pd(a, b);
}

static inline __m256d op_add_relu(__m256d a, __m256d b,
                                  __m256d zero __attribute__((unused))) {
    return _mm256_max_pd(_mm256_add_pd(a, b), zero);
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
static inline void VV_binop(VEC vdest, VEC va, VEC_restr vb, binop_pd_t op) {
    assert(VV_FIT(vdest, va));
    assert(VV_FIT(vdest, vb));

    const size_t  pad_dim = VEC_pad_dim(vdest);
    const __m256d zero    = _mm256_setzero_pd();

    double *ld_a    = va;
    double *ld_b    = vb;
    double *st_dest = vdest;

    size_t i = 0;
    for (; i + 15 < pad_dim; i += 16) {
        __m256d ya[2][2], yb[2][2];
        LOAD_by_8(ya[0], ld_a);
        LOAD_by_8(yb[0], ld_b);

        ya[0][0] = op(ya[0][0], yb[0][0], zero);
        ya[0][1] = op(ya[0][1], yb[0][1], zero);

        STORE_by_8(st_dest, ya[0]);

        LOAD_by_8(ya[1], ld_a);
        LOAD_by_8(yb[1], ld_b);

        ya[1][0] = op(ya[1][0], yb[1][0], zero);
        ya[1][1] = op(ya[1][1], yb[1][1], zero);

        STORE_by_8(st_dest, ya[1]);
    }
    if (i < pad_dim) {  // remains 8
        __m256d ya[2], yb[2];
        LOAD_by_8(ya, ld_a);
        LOAD_by_8(yb, ld_b);

        ya[0] = op(ya[0], yb[0], zero);
        ya[1] = op(ya[1], yb[1], zero);

        STORE_by_8(st_dest, ya);
    }
}

void VV_add(VEC vdest, VEC va, VEC_restr vb) {
    VV_binop(vdest, va, vb, op_add);
}

void VV_add_relu(VEC vdest, VEC va, VEC_restr vb) {
    VV_binop(vdest, va, vb, op_add_relu);
}

void VV_sub(VEC vdest, VEC va, VEC_restr vb) {
    VV_binop(vdest, va, vb, op_sub);
}

void VV_sub_pos(VEC_restr vdest, VEC_restr va) {
    VV_binop(vdest, vdest, va, op_sub_pos);
}

void VV_sub_neg(VEC_restr vdest, VEC_restr va) {
    VV_binop(vdest, vdest, va, op_sub_neg);
}

// ------------------------------- Matrix-Vector -------------------------------
// vdest <- op(op(va, mb[list[0]]), m[list[1]]) ...
static inline void MV_binop(VEC vdest, VEC va, MAT_restr mb, binop_pd_t op,
                            uint16_t *list, size_t len) {
    assert(M2V_FIT(mb, vdest));
    assert(M2V_FIT(mb, va));

    if (list == NULL)
        len = MAT_dim1(mb);

    if (len == 0) {
        if (vdest != va)
            VEC_copy(vdest, va);
        return;
    }

    const size_t  pad_dim = VEC_pad_dim(vdest);
    const __m256d zero    = _mm256_setzero_pd();

    size_t j = 0;
    for (; j + 3 < len; j += 4) {
        double *ld_a = (j == 0) ? va : vdest;
        double *ld_b[4];
        if (list) {
            ld_b[0] = mb[list[j]];
            ld_b[1] = mb[list[j + 1]];
            ld_b[2] = mb[list[j + 2]];
            ld_b[3] = mb[list[j + 3]];
        }
        else {
            ld_b[0] = mb[j];
            ld_b[1] = mb[j + 1];
            ld_b[2] = mb[j + 2];
            ld_b[3] = mb[j + 3];
        }

        double *st_dest = vdest;

        for (size_t i = 0; i < pad_dim; i += 8) {
            __m256d ya[2], yb[4][2];
            LOAD_by_8(ya, ld_a);
            LOAD_by_8(yb[0], ld_b[0]);
            LOAD_by_8(yb[1], ld_b[1]);
            LOAD_by_8(yb[2], ld_b[2]);
            LOAD_by_8(yb[3], ld_b[3]);

            ya[0] = op(ya[0], yb[0][0], zero);
            ya[0] = op(ya[0], yb[1][0], zero);
            ya[0] = op(ya[0], yb[2][0], zero);
            ya[0] = op(ya[0], yb[3][0], zero);
            ya[1] = op(ya[1], yb[0][1], zero);
            ya[1] = op(ya[1], yb[1][1], zero);
            ya[1] = op(ya[1], yb[2][1], zero);
            ya[1] = op(ya[1], yb[3][1], zero);

            STORE_by_8(st_dest, ya);
        }
    }
    if (j + 1 < len) {
        double *ld_a = (j == 0) ? va : vdest;
        double *ld_b[2];
        if (list) {
            ld_b[0] = mb[list[j]];
            ld_b[1] = mb[list[j + 1]];
        }
        else {
            ld_b[0] = mb[j];
            ld_b[1] = mb[j + 1];
        }

        double *st_dest = vdest;

        for (size_t i = 0; i < pad_dim; i += 8) {
            __m256d ya[2], yb[2][2];
            LOAD_by_8(ya, ld_a);
            LOAD_by_8(yb[0], ld_b[0]);
            LOAD_by_8(yb[1], ld_b[1]);

            ya[0] = op(ya[0], yb[0][0], zero);
            ya[0] = op(ya[0], yb[1][0], zero);
            ya[1] = op(ya[1], yb[0][1], zero);
            ya[1] = op(ya[1], yb[1][1], zero);

            STORE_by_8(st_dest, ya);
        }

        j += 2;
    }
    if (j < len)
        VV_binop(vdest, (j == 0) ? va : vdest, mb[lookup(list, j)], op);
}

void MV_acc(VEC vdest, VEC va, MAT_restr mb, uint16_t *list, size_t len) {
    if (va == NULL) {
        VEC_clear(vdest);
        MV_binop(vdest, vdest, mb, op_add, list, len);
    }
    else {
        MV_binop(vdest, va, mb, op_add, list, len);
    }
}

void MV_acc_pos(VEC vdest, VEC va, MAT_restr mb, uint16_t *list, size_t len) {
    if (va == NULL) {
        VEC_clear(vdest);
        MV_binop(vdest, vdest, mb, op_add_pos, list, len);
    }
    else {
        MV_binop(vdest, va, mb, op_add_pos, list, len);
    }
}

void MV_acc_neg(VEC vdest, VEC va, MAT_restr mb, uint16_t *list, size_t len) {
    if (va == NULL) {
        VEC_clear(vdest);
        MV_binop(vdest, vdest, mb, op_add_neg, list, len);
    }
    else {
        MV_binop(vdest, va, mb, op_add_neg, list, len);
    }
}

void MV_max(VEC vdest, VEC va, MAT_restr mb, uint16_t *list, size_t len) {
    if (va == NULL) {
        if (len == 0) {
            VEC_clear(vdest);
        }
        else {
            VEC_set(vdest, -DBL_MAX);
            MV_binop(vdest, vdest, mb, op_max, list, len);
        }
    }
    else {
        MV_binop(vdest, va, mb, op_max, list, len);
    }
}

void MV_min(VEC vdest, VEC va, MAT_restr mb, uint16_t *list, size_t len) {
    if (va == NULL) {
        if (len == 0) {
            VEC_clear(vdest);
        }
        else {
            VEC_set(vdest, DBL_MAX);
            MV_binop(vdest, vdest, mb, op_min, list, len);
        }
    }
    else {
        MV_binop(vdest, va, mb, op_min, list, len);
    }
}

// ------------------------------- Matrix-Matrix -------------------------------
void MM_dot(MAT_restr mdest, MAT_restr ma, MAT_restr mb) {
    assert(M1M1_FIT(mdest, ma));
    assert(M2M2_FIT(mdest, mb));
    assert(M2M1_FIT(ma, mb));

    ITER_MAT1(mdest, i) {
        ITER_MAT2(mdest, j) {
            mdest[i][j] = 0.;
            ITER_MAT2(ma, k) {
                mdest[i][j] += ma[i][k] * mb[k][j];
            }
        }
    }
}

static inline __m256d MV_dot1_4(VEC_restr *ma, VEC_restr vb, size_t pad_dim) {
    __m256d acc[4];

    acc[0] = _mm256_setzero_pd();
    acc[1] = _mm256_setzero_pd();
    acc[2] = _mm256_setzero_pd();
    acc[3] = _mm256_setzero_pd();

    double *ld_b = vb;
    double *ld_a[4];
    ld_a[0] = ma[0];
    ld_a[1] = ma[1];
    ld_a[2] = ma[2];
    ld_a[3] = ma[3];

    for (size_t i = 0; i < pad_dim; i += 8) {
        __m256d yb[2], ya[4][2];
        LOAD_by_8(yb, ld_b);
        LOAD_by_8(ya[0], ld_a[0]);
        LOAD_by_8(ya[1], ld_a[1]);
        LOAD_by_8(ya[2], ld_a[2]);
        LOAD_by_8(ya[3], ld_a[3]);

        acc[0] = _mm256_fmadd_pd(ya[0][0], yb[0], acc[0]);
        acc[1] = _mm256_fmadd_pd(ya[1][0], yb[0], acc[1]);
        acc[2] = _mm256_fmadd_pd(ya[2][0], yb[0], acc[2]);
        acc[3] = _mm256_fmadd_pd(ya[3][0], yb[0], acc[3]);

        acc[0] = _mm256_fmadd_pd(ya[0][1], yb[1], acc[0]);
        acc[1] = _mm256_fmadd_pd(ya[1][1], yb[1], acc[1]);
        acc[2] = _mm256_fmadd_pd(ya[2][1], yb[1], acc[2]);
        acc[3] = _mm256_fmadd_pd(ya[3][1], yb[1], acc[3]);
    }

    return hsum4(acc);
}

static inline double MV_dot1_1(VEC_restr va, VEC_restr vb, size_t pad_dim) {
    __m256d acc = _mm256_setzero_pd();

    double *ld_b = vb;
    double *ld_a = va;

    for (size_t i = 0; i < pad_dim; i += 8) {
        __m256d yb[2], ya[2];
        LOAD_by_8(yb, ld_b);
        LOAD_by_8(ya, ld_a);

        acc = _mm256_fmadd_pd(ya[0], yb[0], acc);
        acc = _mm256_fmadd_pd(ya[1], yb[1], acc);
    }

    return hsum1(acc);
}

static void MV_dot1(VEC_restr vdest, MAT_restr ma, VEC_restr vb, VEC_restr vc) {
    assert(M1V_FIT(ma, vdest));
    assert(M2V_FIT(ma, vb));

    const size_t dest_dim = VEC_dim(vdest);
    const size_t pad_dim  = VEC_pad_dim(vb);

    double *st_dest = vdest;

    if (vc == NULL) {
        size_t j = 0;
        for (; j + 3 < dest_dim; j += 4) {
            __m256d val = MV_dot1_4(ma + j, vb, pad_dim);
            _mm256_store_pd(st_dest, val);
            st_dest += 4;
        }
        for (; j < dest_dim; ++j) {
            *st_dest = MV_dot1_1(ma[j], vb, pad_dim);
            st_dest++;
        }
    }
    else {
        size_t j = 0;
        for (; j + 3 < dest_dim; j += 4) {
            __m256d val = MV_dot1_4(ma + j, vb, pad_dim);
            __m256d yc  = _mm256_load_pd(vc + j);
            _mm256_store_pd(st_dest, _mm256_add_pd(val, yc));
            st_dest += 4;
        }
        for (; j < dest_dim; ++j) {
            *st_dest = MV_dot1_1(ma[j], vb, pad_dim) + vc[j];
            st_dest++;
        }
    }
}

static inline void MV_dot2_4(__m256d val[2], VEC_restr *ma, VEC_restr vb[2],
                             size_t pad_dim) {
    __m256d acc0[4], acc1[4];
    acc0[0] = _mm256_setzero_pd();
    acc0[1] = _mm256_setzero_pd();
    acc0[2] = _mm256_setzero_pd();
    acc0[3] = _mm256_setzero_pd();
    acc1[0] = _mm256_setzero_pd();
    acc1[1] = _mm256_setzero_pd();
    acc1[2] = _mm256_setzero_pd();
    acc1[3] = _mm256_setzero_pd();

    double *ld_b0 = vb[0];
    double *ld_b1 = vb[1];
    double *ld_a[4];
    ld_a[0] = ma[0];
    ld_a[1] = ma[1];
    ld_a[2] = ma[2];
    ld_a[3] = ma[3];

    for (size_t i = 0; i < pad_dim; i += 8) {
        __m256d yb0[2], yb1[2], ya[4][2];
        LOAD_by_8(yb0, ld_b0);
        LOAD_by_8(yb1, ld_b1);
        LOAD_by_8(ya[0], ld_a[0]);
        LOAD_by_8(ya[1], ld_a[1]);
        LOAD_by_8(ya[2], ld_a[2]);
        LOAD_by_8(ya[3], ld_a[3]);

        acc0[0] = _mm256_fmadd_pd(ya[0][0], yb0[0], acc0[0]);
        acc0[1] = _mm256_fmadd_pd(ya[1][0], yb0[0], acc0[1]);
        acc0[2] = _mm256_fmadd_pd(ya[2][0], yb0[0], acc0[2]);
        acc0[3] = _mm256_fmadd_pd(ya[3][0], yb0[0], acc0[3]);
        acc0[0] = _mm256_fmadd_pd(ya[0][1], yb0[1], acc0[0]);
        acc0[1] = _mm256_fmadd_pd(ya[1][1], yb0[1], acc0[1]);
        acc0[2] = _mm256_fmadd_pd(ya[2][1], yb0[1], acc0[2]);
        acc0[3] = _mm256_fmadd_pd(ya[3][1], yb0[1], acc0[3]);

        acc1[0] = _mm256_fmadd_pd(ya[0][0], yb1[0], acc1[0]);
        acc1[1] = _mm256_fmadd_pd(ya[1][0], yb1[0], acc1[1]);
        acc1[2] = _mm256_fmadd_pd(ya[2][0], yb1[0], acc1[2]);
        acc1[3] = _mm256_fmadd_pd(ya[3][0], yb1[0], acc1[3]);
        acc1[0] = _mm256_fmadd_pd(ya[0][1], yb1[1], acc1[0]);
        acc1[1] = _mm256_fmadd_pd(ya[1][1], yb1[1], acc1[1]);
        acc1[2] = _mm256_fmadd_pd(ya[2][1], yb1[1], acc1[2]);
        acc1[3] = _mm256_fmadd_pd(ya[3][1], yb1[1], acc1[3]);
    }

    val[0] = hsum4(acc0);
    val[1] = hsum4(acc1);
}

static inline __m128d MV_dot2_1(VEC_restr va, VEC_restr vb[2], size_t pad_dim) {
    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();

    double *ld_b0 = vb[0];
    double *ld_b1 = vb[1];
    double *ld_a  = va;

    for (size_t i = 0; i < pad_dim; i += 8) {
        __m256d yb0[2], yb1[2], ya[2];
        LOAD_by_8(yb0, ld_b0);
        LOAD_by_8(yb1, ld_b1);
        LOAD_by_8(ya, ld_a);

        acc0 = _mm256_fmadd_pd(ya[0], yb0[0], acc0);
        acc0 = _mm256_fmadd_pd(ya[1], yb0[1], acc0);
        acc1 = _mm256_fmadd_pd(ya[0], yb1[0], acc1);
        acc1 = _mm256_fmadd_pd(ya[1], yb1[1], acc1);
    }

    return hsum2(acc0, acc1);
}

static void MV_dot2(VEC_restr vdest[2], MAT_restr ma, VEC_restr vb[2],
                    VEC_restr vc) {
    assert(M1V_FIT(ma, vdest[0]));
    assert(M1V_FIT(ma, vdest[1]));
    assert(M2V_FIT(ma, vb[0]));
    assert(M2V_FIT(ma, vb[1]));

    const size_t dest_dim = VEC_dim(vdest[0]);
    const size_t pad_dim  = VEC_pad_dim(vb[0]);

    double *st_dest0 = vdest[0];
    double *st_dest1 = vdest[1];

    if (vc == NULL) {
        size_t j = 0;
        for (; j + 3 < dest_dim; j += 4) {
            __m256d val[2];
            MV_dot2_4(val, ma + j, vb, pad_dim);
            _mm256_store_pd(st_dest0, val[0]);
            _mm256_store_pd(st_dest1, val[1]);
            st_dest0 += 4;
            st_dest1 += 4;
        }
        for (; j < dest_dim; ++j) {
            double val[2];
            _mm_store_pd(val, MV_dot2_1(ma[j], vb, pad_dim));
            *st_dest0 = val[0];
            *st_dest1 = val[1];
            st_dest0++;
            st_dest1++;
        }
    }
    else {
        size_t j = 0;
        for (; j + 3 < dest_dim; j += 4) {
            __m256d val[2];
            MV_dot2_4(val, ma + j, vb, pad_dim);
            __m256d yc = _mm256_load_pd(vc + j);
            _mm256_store_pd(st_dest0, _mm256_add_pd(val[0], yc));
            _mm256_store_pd(st_dest1, _mm256_add_pd(val[1], yc));
            st_dest0 += 4;
            st_dest1 += 4;
        }
        for (; j < dest_dim; ++j) {
            double val[2];
            _mm_store_pd(val, _mm_add_pd(MV_dot2_1(ma[j], vb, pad_dim),
                                         _mm_set1_pd(vc[j])));
            *st_dest0 = val[0];
            *st_dest1 = val[1];
            st_dest0++;
            st_dest1++;
        }
    }
}

void MV_dot(VEC_restr vdest, MAT_restr ma, VEC_restr vb, VEC_restr vc) {
    assert(M1V_FIT(ma, vdest));
    assert(M2V_FIT(ma, vb));

    MV_dot1(vdest, ma, vb, vc);
}

void MM_dot_trans(MAT_restr mdest, MAT_restr ma, MAT_restr mb, VEC_restr vc,
                  uint16_t *list, size_t len) {
    assert(M1M1_FIT(mdest, ma));
    assert(M2M1_FIT(mdest, mb));
    assert(M2M2_FIT(ma, mb));

    len = list ? len : MAT_dim1(ma);

    size_t i = 0;
    for (; i + 1 < len; i += 2) {
        size_t index0 = lookup(list, i);
        size_t index1 = lookup(list, i + 1);

        VEC vdest[2] = {mdest[index0], mdest[index1]};
        VEC va[2]    = {ma[index0], ma[index1]};

        MV_dot2(vdest, mb, va, vc);
    }
    if (i < len) {  // remains
        size_t index = lookup(list, i);

        MV_dot1(mdest[index], mb, ma[index], vc);
    }
}

void MM_relax_trans(MAT_restr mdestl, MAT_restr mdestu, MAT_restr mal,
                    MAT_restr mau, MAT_restr mb, VEC_restr vc, MAT_restr mtmps,
                    MAT_restr mtmpd, uint16_t *list, size_t len) {
    assert(M1M1_FIT(mdestl, mdestu));
    assert(M2M2_FIT(mdestl, mdestu));
    assert(M1M1_FIT(mal, mau));
    assert(M2M2_FIT(mal, mau));
    assert(M1M1_FIT(mtmps, mtmpd));
    assert(M2M2_FIT(mtmps, mtmpd));

    assert(M1M1_FIT(mal, mtmps));
    assert(M2M2_FIT(mal, mtmps));

    assert(M1M1_FIT(mdestl, mal));
    assert(M2M1_FIT(mdestl, mb));
    assert(M2M2_FIT(mal, mb));

    len = list ? len : MAT_dim1(mal);

    const size_t pad_dim  = VEC_pad_dim(mal[0]);
    const size_t dest_dim = VEC_dim(mdestl[0]);

    const __m256d half = _mm256_set1_pd(.5);

    for (size_t j = 0; j < len; ++j) {
        size_t  index = lookup(list, j);
        double *ld_al = mal[index];
        double *ld_au = mau[index];
        double *st_s  = mtmps[j];
        double *st_d  = mtmpd[j];

        for (size_t i = 0; i < pad_dim; i += 8) {
            __m256d yal[2], yau[2], ys[2], yd[2];
            LOAD_by_8(yal, ld_al);
            LOAD_by_8(yau, ld_au);

            ys[0] = _mm256_mul_pd(_mm256_add_pd(yau[0], yal[0]), half);
            yd[0] = _mm256_mul_pd(_mm256_sub_pd(yau[0], yal[0]), half);
            ys[1] = _mm256_mul_pd(_mm256_add_pd(yau[1], yal[1]), half);
            yd[1] = _mm256_mul_pd(_mm256_sub_pd(yau[1], yal[1]), half);

            STORE_by_8(st_s, ys);
            STORE_by_8(st_d, yd);
        }
    }

    MAT_restr ms = mb;
    MAT_restr md = MAT_abs(mb);

    if (vc == NULL) {
        size_t k = 0;
        for (; k + 1 < len; k += 2) {
            size_t index0 = lookup(list, k);
            size_t index1 = lookup(list, k + 1);

            double *st_l0 = mdestl[index0];
            double *st_l1 = mdestl[index1];
            double *st_u0 = mdestu[index0];
            double *st_u1 = mdestu[index1];

            size_t j = 0;
            for (; j + 3 < dest_dim; j += 4) {
                __m256d ys[2], yd[2];
                MV_dot2_4(ys, ms + j, mtmps + k, pad_dim);
                MV_dot2_4(yd, md + j, mtmpd + k, pad_dim);
                _mm256_store_pd(st_l0, _mm256_sub_pd(ys[0], yd[0]));
                _mm256_store_pd(st_l1, _mm256_sub_pd(ys[1], yd[1]));
                _mm256_store_pd(st_u0, _mm256_add_pd(ys[0], yd[0]));
                _mm256_store_pd(st_u1, _mm256_add_pd(ys[1], yd[1]));
                st_l0 += 4;
                st_l1 += 4;
                st_u0 += 4;
                st_u1 += 4;
            }
            for (; j < dest_dim; ++j) {
                __m128d ys = MV_dot2_1(ms[j], mtmps + k, pad_dim);
                __m128d yd = MV_dot2_1(md[j], mtmpd + k, pad_dim);
                __m128d yl = _mm_sub_pd(ys, yd);
                __m128d yu = _mm_add_pd(ys, yd);
                _mm_store_sd(st_l0, yl);
                _mm_storeh_pd(st_l1, yl);
                _mm_store_sd(st_u0, yu);
                _mm_storeh_pd(st_u1, yu);
                st_l0++;
                st_l1++;
                st_u0++;
                st_u1++;
            }
        }
        if (k < len) {  // remains
            size_t index = lookup(list, k);

            double *st_l = mdestl[index];
            double *st_u = mdestu[index];

            size_t j = 0;
            for (; j + 3 < dest_dim; j += 4) {
                __m256d ys = MV_dot1_4(ms + j, mtmps[k], pad_dim);
                __m256d yd = MV_dot1_4(md + j, mtmpd[k], pad_dim);
                _mm256_store_pd(st_l, _mm256_sub_pd(ys, yd));
                _mm256_store_pd(st_u, _mm256_add_pd(ys, yd));
                st_l += 4;
                st_u += 4;
            }
            for (; j < dest_dim; ++j) {
                double vals = MV_dot1_1(ms[j], mtmps[k], pad_dim);
                double vald = MV_dot1_1(md[j], mtmpd[k], pad_dim);
                *st_l       = vals - vald;
                *st_u       = vals + vald;
                st_l++;
                st_u++;
            }
        }
    }
    else {
        size_t k = 0;
        for (; k + 1 < len; k += 2) {
            size_t index0 = lookup(list, k);
            size_t index1 = lookup(list, k + 1);

            double *st_l0 = mdestl[index0];
            double *st_l1 = mdestl[index1];
            double *st_u0 = mdestu[index0];
            double *st_u1 = mdestu[index1];

            size_t j = 0;
            for (; j + 3 < dest_dim; j += 4) {
                __m256d ys[2], yd[2];
                MV_dot2_4(ys, ms + j, mtmps + k, pad_dim);
                MV_dot2_4(yd, md + j, mtmpd + k, pad_dim);
                __m256d yc = _mm256_load_pd(vc + j);
                ys[0]      = _mm256_add_pd(ys[0], yc);
                ys[1]      = _mm256_add_pd(ys[1], yc);
                _mm256_store_pd(st_l0, _mm256_sub_pd(ys[0], yd[0]));
                _mm256_store_pd(st_l1, _mm256_sub_pd(ys[1], yd[1]));
                _mm256_store_pd(st_u0, _mm256_add_pd(ys[0], yd[0]));
                _mm256_store_pd(st_u1, _mm256_add_pd(ys[1], yd[1]));
                st_l0 += 4;
                st_l1 += 4;
                st_u0 += 4;
                st_u1 += 4;
            }
            for (; j < dest_dim; ++j) {
                __m128d ys = MV_dot2_1(ms[j], mtmps + k, pad_dim);
                __m128d yd = MV_dot2_1(md[j], mtmpd + k, pad_dim);
                ys         = _mm_add_pd(ys, _mm_set1_pd(vc[j]));
                __m128d yl = _mm_sub_pd(ys, yd);
                __m128d yu = _mm_add_pd(ys, yd);
                _mm_store_sd(st_l0, yl);
                _mm_storeh_pd(st_l1, yl);
                _mm_store_sd(st_u0, yu);
                _mm_storeh_pd(st_u1, yu);
                st_l0++;
                st_l1++;
                st_u0++;
                st_u1++;
            }
        }
        if (k < len) {  // remains
            size_t index = lookup(list, k);

            double *st_l = mdestl[index];
            double *st_u = mdestu[index];

            size_t j = 0;
            for (; j + 3 < dest_dim; j += 4) {
                __m256d ys = MV_dot1_4(ms + j, mtmps[k], pad_dim);
                __m256d yd = MV_dot1_4(md + j, mtmpd[k], pad_dim);
                ys         = _mm256_add_pd(ys, _mm256_load_pd(vc + j));
                _mm256_store_pd(st_l, _mm256_sub_pd(ys, yd));
                _mm256_store_pd(st_u, _mm256_add_pd(ys, yd));
                st_l += 4;
                st_u += 4;
            }
            for (; j < dest_dim; ++j) {
                double vals = MV_dot1_1(ms[j], mtmps[k], pad_dim);
                double vald = MV_dot1_1(md[j], mtmpd[k], pad_dim);
                vals += vc[j];
                *st_l = vals - vald;
                *st_u = vals + vald;
                st_l++;
                st_u++;
            }
        }
    }
}

// ---------------------------------- sorted -----------------------------------
void MV_trinity(VEC_restr vacc, VEC_restr vmax, VEC_restr vmin, MAT_restr ma,
                uint16_t *list, size_t len) {
    assert(VV_FIT(vacc, vmax));
    assert(VV_FIT(vacc, vmin));
    assert(M2V_FIT(ma, vacc));

    const size_t pad_dim = VEC_pad_dim(vacc);

    size_t j = 0;
    for (; j + 3 < len; j += 4) {
        double *ld_acc = vacc;
        double *ld_max = vmax;
        double *ld_min = vmin;
        double *ld_a[4];
        ld_a[0]        = ma[list[j]];
        ld_a[1]        = ma[list[j + 1]];
        ld_a[2]        = ma[list[j + 2]];
        ld_a[3]        = ma[list[j + 3]];
        double *st_acc = vacc;
        double *st_max = vmax;
        double *st_min = vmin;

        for (size_t i = 0; i < pad_dim; i += 8) {
            __m256d yacc[2], ymax[2], ymin[2], ya[4][2];
            LOAD_by_8(yacc, ld_acc);
            if (j == 0) {
                ymax[0] = _mm256_set1_pd(-DBL_MAX);
                ymax[1] = _mm256_set1_pd(-DBL_MAX);
                ymin[0] = _mm256_set1_pd(DBL_MAX);
                ymin[1] = _mm256_set1_pd(DBL_MAX);
            }
            else {
                LOAD_by_8(ymax, ld_max);
                LOAD_by_8(ymin, ld_min);
            }
            LOAD_by_8(ya[0], ld_a[0]);
            LOAD_by_8(ya[1], ld_a[1]);
            LOAD_by_8(ya[2], ld_a[2]);
            LOAD_by_8(ya[3], ld_a[3]);

            yacc[0] = _mm256_add_pd(yacc[0], ya[0][0]);
            yacc[0] = _mm256_add_pd(yacc[0], ya[1][0]);
            yacc[0] = _mm256_add_pd(yacc[0], ya[2][0]);
            yacc[0] = _mm256_add_pd(yacc[0], ya[3][0]);

            ymax[0] = _mm256_max_pd(ymax[0], ya[0][0]);
            ymax[0] = _mm256_max_pd(ymax[0], ya[1][0]);
            ymax[0] = _mm256_max_pd(ymax[0], ya[2][0]);
            ymax[0] = _mm256_max_pd(ymax[0], ya[3][0]);

            ymin[0] = _mm256_min_pd(ymin[0], ya[0][0]);
            ymin[0] = _mm256_min_pd(ymin[0], ya[1][0]);
            ymin[0] = _mm256_min_pd(ymin[0], ya[2][0]);
            ymin[0] = _mm256_min_pd(ymin[0], ya[3][0]);

            yacc[1] = _mm256_add_pd(yacc[1], ya[0][1]);
            yacc[1] = _mm256_add_pd(yacc[1], ya[1][1]);
            yacc[1] = _mm256_add_pd(yacc[1], ya[2][1]);
            yacc[1] = _mm256_add_pd(yacc[1], ya[3][1]);

            ymax[1] = _mm256_max_pd(ymax[1], ya[0][1]);
            ymax[1] = _mm256_max_pd(ymax[1], ya[1][1]);
            ymax[1] = _mm256_max_pd(ymax[1], ya[2][1]);
            ymax[1] = _mm256_max_pd(ymax[1], ya[3][1]);

            ymin[1] = _mm256_min_pd(ymin[1], ya[0][1]);
            ymin[1] = _mm256_min_pd(ymin[1], ya[1][1]);
            ymin[1] = _mm256_min_pd(ymin[1], ya[2][1]);
            ymin[1] = _mm256_min_pd(ymin[1], ya[3][1]);

            STORE_by_8(st_acc, yacc);
            STORE_by_8(st_max, ymax);
            STORE_by_8(st_min, ymin);
        }
    }
    for (; j < len; ++j) {
        double *ld_a   = ma[list[j]];
        double *ld_acc = vacc;
        double *ld_max = vmax;
        double *ld_min = vmin;
        double *st_acc = vacc;
        double *st_max = vmax;
        double *st_min = vmin;

        for (size_t i = 0; i < pad_dim; i += 8) {
            __m256d yacc[2], ymax[2], ymin[2], ya[2];
            LOAD_by_8(yacc, ld_acc);
            if (j == 0) {
                ymax[0] = _mm256_set1_pd(-DBL_MAX);
                ymax[1] = _mm256_set1_pd(-DBL_MAX);
                ymin[0] = _mm256_set1_pd(DBL_MAX);
                ymin[1] = _mm256_set1_pd(DBL_MAX);
            }
            else {
                LOAD_by_8(ymax, ld_max);
                LOAD_by_8(ymin, ld_min);
            }
            LOAD_by_8(ya, ld_a);

            yacc[0] = _mm256_add_pd(yacc[0], ya[0]);
            ymax[0] = _mm256_max_pd(ymax[0], ya[0]);
            ymin[0] = _mm256_min_pd(ymin[0], ya[0]);

            yacc[1] = _mm256_add_pd(yacc[1], ya[1]);
            ymax[1] = _mm256_max_pd(ymax[1], ya[1]);
            ymin[1] = _mm256_min_pd(ymin[1], ya[1]);

            STORE_by_8(st_acc, yacc);
            STORE_by_8(st_max, ymax);
            STORE_by_8(st_min, ymin);
        }
    }
}

static void VV_top_2_of_3(VEC_restr vdest0, VEC_restr vdest1, VEC_restr va) {
    assert(VV_FIT(vdest0, vdest1));
    assert(VV_FIT(vdest0, va));

    const size_t pad_dim = VEC_pad_dim(va);
    __m256d      tmp;

    double *ld_dest0 = vdest0;
    double *ld_dest1 = vdest1;
    double *ld_a     = va;
    double *st_dest0 = vdest0;
    double *st_dest1 = vdest1;

    for (size_t i = 0; i < pad_dim; i += 8) {
        __m256d ydest0[2], ydest1[2], ya[2];
        LOAD_by_8(ydest0, ld_dest0);
        LOAD_by_8(ydest1, ld_dest1);
        LOAD_by_8(ya, ld_a);

        CSWAP(ydest0[0], ya[0]);
        CSWAP(ydest1[0], ya[0]);

        CSWAP(ydest0[1], ya[1]);
        CSWAP(ydest1[1], ya[1]);

        STORE_by_8(st_dest0, ydest0);
        STORE_by_8(st_dest1, ydest1);
    }
}

static void VV_bot_2_of_3(VEC_restr vdest0, VEC_restr vdest1, VEC_restr va) {
    assert(VV_FIT(vdest0, vdest1));
    assert(VV_FIT(vdest0, va));

    const size_t pad_dim = VEC_pad_dim(va);
    __m256d      tmp;

    double *ld_dest0 = vdest0;
    double *ld_dest1 = vdest1;
    double *ld_a     = va;
    double *st_dest0 = vdest0;
    double *st_dest1 = vdest1;

    for (size_t i = 0; i < pad_dim; i += 8) {
        __m256d ydest0[2], ydest1[2], ya[2];
        LOAD_by_8(ydest0, ld_dest0);
        LOAD_by_8(ydest1, ld_dest1);
        LOAD_by_8(ya, ld_a);

        CSWAP(ya[0], ydest0[0]);
        CSWAP(ya[0], ydest1[0]);

        CSWAP(ya[1], ydest0[1]);
        CSWAP(ya[1], ydest1[1]);

        STORE_by_8(st_dest0, ydest0);
        STORE_by_8(st_dest1, ydest1);
    }
}

static void VV_top_2_of_4(VEC_restr vdest0, VEC_restr vdest1, VEC_restr va,
                          VEC_restr vb) {
    assert(VV_FIT(vdest0, vdest1));
    assert(VV_FIT(vdest0, va));
    assert(VV_FIT(vdest0, vb));

    const size_t pad_dim = VEC_pad_dim(va);
    __m256d      tmp;

    double *ld_dest0 = vdest0;
    double *ld_dest1 = vdest1;
    double *ld_a     = va;
    double *ld_b     = vb;
    double *st_dest0 = vdest0;
    double *st_dest1 = vdest1;

    for (size_t i = 0; i < pad_dim; i += 8) {
        __m256d ydest0[2], ydest1[2], ya[2], yb[2];
        LOAD_by_8(ydest0, ld_dest0);
        LOAD_by_8(ydest1, ld_dest1);
        LOAD_by_8(ya, ld_a);
        LOAD_by_8(yb, ld_b);

        CSWAP(ydest0[0], ya[0]);
        CSWAP(ydest0[0], yb[0]);
        CSWAP(ydest1[0], ya[0]);
        CSWAP(ydest1[0], yb[0]);

        CSWAP(ydest0[1], ya[1]);
        CSWAP(ydest0[1], yb[1]);
        CSWAP(ydest1[1], ya[1]);
        CSWAP(ydest1[1], yb[1]);

        STORE_by_8(st_dest0, ydest0);
        STORE_by_8(st_dest1, ydest1);
    }
}

static void VV_bot_2_of_4(VEC_restr vdest0, VEC_restr vdest1, VEC_restr va,
                          VEC_restr vb) {
    assert(VV_FIT(vdest0, vdest1));
    assert(VV_FIT(vdest0, va));
    assert(VV_FIT(vdest0, vb));

    const size_t pad_dim = VEC_pad_dim(va);
    __m256d      tmp;

    double *ld_dest0 = vdest0;
    double *ld_dest1 = vdest1;
    double *ld_a     = va;
    double *ld_b     = vb;
    double *st_dest0 = vdest0;
    double *st_dest1 = vdest1;

    for (size_t i = 0; i < pad_dim; i += 8) {
        __m256d ydest0[2], ydest1[2], ya[2], yb[2];
        LOAD_by_8(ydest0, ld_dest0);
        LOAD_by_8(ydest1, ld_dest1);
        LOAD_by_8(ya, ld_a);
        LOAD_by_8(yb, ld_b);

        CSWAP(ya[0], ydest0[0]);
        CSWAP(yb[0], ydest0[0]);
        CSWAP(ya[0], ydest1[0]);
        CSWAP(yb[0], ydest1[0]);

        CSWAP(ya[1], ydest0[1]);
        CSWAP(yb[1], ydest0[1]);
        CSWAP(ya[1], ydest1[1]);
        CSWAP(yb[1], ydest1[1]);

        STORE_by_8(st_dest0, ydest0);
        STORE_by_8(st_dest1, ydest1);
    }
}

static void VV_bot_3_of_4(VEC_restr vdest0, VEC_restr vdest1, VEC_restr vdest2,
                          VEC_restr va) {
    assert(VV_FIT(vdest0, vdest1));
    assert(VV_FIT(vdest0, vdest2));
    assert(VV_FIT(vdest0, va));

    const size_t pad_dim = VEC_pad_dim(va);
    __m256d      tmp;

    double *ld_dest0 = vdest0;
    double *ld_dest1 = vdest1;
    double *ld_dest2 = vdest2;
    double *ld_a     = va;
    double *st_dest0 = vdest0;
    double *st_dest1 = vdest1;
    double *st_dest2 = vdest2;

    for (size_t i = 0; i < pad_dim; i += 8) {
        __m256d ydest0[2], ydest1[2], ydest2[2], ya[2];
        LOAD_by_8(ydest0, ld_dest0);
        LOAD_by_8(ydest1, ld_dest1);
        LOAD_by_8(ydest2, ld_dest2);
        LOAD_by_8(ya, ld_a);

        CSWAP(ya[0], ydest0[0]);
        CSWAP(ya[0], ydest1[0]);
        CSWAP(ya[0], ydest2[0]);

        CSWAP(ya[1], ydest0[1]);
        CSWAP(ya[1], ydest1[1]);
        CSWAP(ya[1], ydest2[1]);

        STORE_by_8(st_dest0, ydest0);
        STORE_by_8(st_dest1, ydest1);
        STORE_by_8(st_dest2, ydest2);
    }
}

void MV_max2(VEC_restr vdest0, VEC_restr vdest1, MAT_restr ma, uint16_t *list,
             size_t len) {
    assert(VV_FIT(vdest0, vdest1));
    assert(M2V_FIT(ma, vdest0));

    VEC_set(vdest0, -DBL_MAX);
    VEC_set(vdest1, -DBL_MAX);

    size_t i = 0;
    for (; i + 1 < len; i += 2)
        VV_top_2_of_4(vdest0, vdest1, ma[list[i]], ma[list[i + 1]]);
    for (; i < len; ++i)
        VV_top_2_of_3(vdest0, vdest1, ma[list[i]]);
}

void MV_min2(VEC_restr vdest0, VEC_restr vdest1, MAT_restr ma, uint16_t *list,
             size_t len) {
    assert(VV_FIT(vdest0, vdest1));
    assert(M2V_FIT(ma, vdest0));

    VEC_set(vdest0, DBL_MAX);
    VEC_set(vdest1, DBL_MAX);

    size_t i = 0;
    for (; i + 1 < len; i += 2)
        VV_bot_2_of_4(vdest0, vdest1, ma[list[i]], ma[list[i + 1]]);
    for (; i < len; ++i)
        VV_bot_2_of_3(vdest0, vdest1, ma[list[i]]);
}

void MV_min3(VEC_restr vdest0, VEC_restr vdest1, VEC_restr vdest2, MAT_restr ma,
             uint16_t *list, size_t len) {
    assert(VV_FIT(vdest0, vdest1));
    assert(VV_FIT(vdest0, vdest2));
    assert(M2V_FIT(ma, vdest0));

    VEC_set(vdest0, DBL_MAX);
    VEC_set(vdest1, DBL_MAX);
    VEC_set(vdest2, DBL_MAX);

    size_t i = 0;
    for (; i < len; ++i)
        VV_bot_3_of_4(vdest0, vdest1, vdest2, ma[list[i]]);
}
