#pragma once

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

// ---------------------------------- macros -----------------------------------
#define MAX(x, y) \
    (((x) >= (y)) ? (x) : (y))

#define MIN(x, y) \
    (((x) <= (y)) ? (x) : (y))

#define ABS(x) \
    (((x) >= 0.) ? (x) : -(x))

#define PAD(x, n) \
    (((x) + ((n)-1)) & ~((n)-1))

#define CHECK_AT(cond, file, line, fmt, ...)               \
    do {                                                   \
        if (cond) {                                        \
            fprintf(stderr, "[RobLight] %s:%d: " fmt "\n", \
                    file, line, ##__VA_ARGS__);            \
            abort();                                       \
        }                                                  \
    } while (0)

#define CHECK(cond, fmt, ...) \
    CHECK_AT(cond, __FILE__, __LINE__, fmt, ##__VA_ARGS__)

static inline void *xmalloc(size_t bytes, const char *file, int line) {
    void *ptr = malloc(bytes);
    CHECK_AT(ptr == NULL, file, line, "malloc failed!");

    return ptr;
}

#define XMALLOC(bytes) \
    xmalloc(bytes, __FILE__, __LINE__)

static inline FILE *xfopen(const char *filename, const char *modes,
                           const char *file, int line) {
    FILE *fp = fopen(filename, modes);
    CHECK_AT(fp == NULL, file, line, "%s do not exist!", filename);

    return fp;
}

#define XFOPEN(filename, mode) \
    xfopen(filename, mode, __FILE__, __LINE__)

// --------------------------------- profiling ---------------------------------
#define PROFILING_START \
    clock_t stamp = (self->profiling) ? clock() : 0

#define PROFILING_END \
    self->clock += (self->profiling) ? (clock() - stamp) : 0

// ----------------------------------- sort ------------------------------------
void qselect(double *a, size_t n, size_t k);
void qsortd(double *a, size_t n);
