#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// ----------------------------------- type ------------------------------------
typedef struct {
    size_t   size;
    size_t   n_chunks;
    uint8_t *data;

    size_t    counter;
    uint16_t *list;

    bool is_dirty_counter;
    bool is_dirty_list;
} BArray;

#define IN_RANGE(self, index) \
    assert((index) < (self)->size)

// ------------------------------- lookup table --------------------------------
extern const uint8_t BArray_tb_test[256][8];
extern const uint8_t BArray_tb_untest[256][8];
extern const uint8_t BArray_tb_byte_set[256][8];
extern const uint8_t BArray_tb_byte_unset[256][8];
extern const uint8_t BArray_tb_count[256];
extern const uint8_t BArray_tb_index[256][8];

// --------------------------------- lifecycle ---------------------------------
BArray *BArray_alloc(size_t size);
void BArray_free(BArray *self);
void BArray_dump(BArray *self);

// ----------------------------- basic operations ------------------------------
static inline bool BArray_test(BArray *self, size_t index) {
    IN_RANGE(self, index);

    size_t byte = index >> 3;
    size_t bit  = index & 7;
    size_t val  = self->data[byte];
    return BArray_tb_test[val][bit];
}

static inline void BArray_set(BArray *self, size_t index) {
    IN_RANGE(self, index);

    size_t byte = index >> 3;
    size_t bit  = index & 7;
    size_t val  = self->data[byte];
    self->counter += BArray_tb_untest[val][bit];
    self->data[byte] = BArray_tb_byte_set[val][bit];

    self->is_dirty_list = true;
}

static inline void BArray_unset(BArray *self, size_t index) {
    IN_RANGE(self, index);

    size_t byte = index >> 3;
    size_t bit  = index & 7;
    size_t val  = self->data[byte];
    self->counter -= BArray_tb_test[val][bit];
    self->data[byte] = BArray_tb_byte_unset[val][bit];

    self->is_dirty_list = true;
}

// --------------------------------- iterator ----------------------------------
size_t BArray_update(BArray *self);

static inline size_t BArray_counter(BArray *self) {
    if (self->is_dirty_counter)
        BArray_update(self);

    return self->counter;
}

static inline uint16_t *BArray_list(BArray *self) {
    BArray_update(self);

    return self->list;
}

#define ITER_BArray(self, item)                                        \
    for (size_t item##_ = 0, item = BArray_update(self),               \
         item##counter = self->counter;                                \
         (item##_ < item##counter) && (item = self->list[item##_], 1); \
         ++item##_)

// -------------------------------- operations ---------------------------------
void BArray_clear(BArray *self);
void BArray_set_all(BArray *self);
void BArray_copy(BArray *self, BArray *ba);
void BArray_union(BArray *self, BArray *ba);
