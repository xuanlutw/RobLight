#include <stddef.h>
#include <stdio.h>

#include "utils.h"

#define CUTOFF_SELECT 4
#define CUTOFF_SORT 16

static inline void dswap(double *a, double *b) {
    double tmp = *a;
    *a         = *b;
    *b         = tmp;
}

static size_t median_of_three(double *a, size_t i, size_t j, size_t k) {
    if (a[i] < a[j]) {
        if (a[j] < a[k])
            return j;
        else if (a[i] < a[k])
            return k;
        else
            return i;
    }
    else {
        if (a[i] < a[k])
            return i;
        else if (a[j] < a[k])
            return k;
        else
            return j;
    }
}

static void insertion_sort(double *a, size_t lo, size_t hi) {
    for (size_t i = lo + 1; i <= hi; ++i) {
        double key = a[i];
        size_t j   = i;
        while (j > lo && a[j - 1] > key) {
            a[j] = a[j - 1];
            j--;
        }
        a[j] = key;
    }
}

static void partition(double *a, size_t lo, size_t hi, size_t *lt, size_t *gt) {
    size_t mid       = lo + (hi - lo) / 2;
    size_t pivot_idx = median_of_three(a, lo, mid, hi);
    double pivot     = a[pivot_idx];

    size_t i = lo;
    *lt      = lo;
    *gt      = hi;

    while (i <= *gt) {
        if (a[i] < pivot) {
            dswap(a + i, a + *lt);
            (*lt)++;
            i++;
        }
        else if (a[i] > pivot) {
            dswap(a + i, a + *gt);
            (*gt)--;
        }
        else
            i++;
    }
}

void qselect(double *a, size_t n, size_t k) {
    size_t lo = 0;
    size_t hi = n - 1;
    size_t lt;
    size_t gt;
    while (lo < hi) {
        if (hi - lo + 1 <= CUTOFF_SELECT) {
            insertion_sort(a, lo, hi);
            return;
        }

        partition(a, lo, hi, &lt, &gt);
        if (k < lt)
            hi = lt - 1;
        else if (k > gt)
            lo = gt + 1;
        else
            break;
    }
}

static void qsortd_(double *a, size_t lo, size_t hi) {
    size_t lt;
    size_t gt;
    while (lo < hi) {
        if (hi - lo + 1 <= CUTOFF_SORT) {
            insertion_sort(a, lo, hi);
            return;
        }

        partition(a, lo, hi, &lt, &gt);
        if (lt == lo)
            lo = gt + 1;
        else if (gt == hi)
            hi = lt - 1;
        else if ((lt - lo) > (hi - gt)) {
            qsortd_(a, gt + 1, hi);
            hi = lt - 1;
        }
        else {
            qsortd_(a, lo, lt - 1);
            lo = gt + 1;
        }
    }
}

void qsortd(double *a, size_t n) {
    if (n <= 1)
        return;

    qsortd_(a, 0, n - 1);
}
