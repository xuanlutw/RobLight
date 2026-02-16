#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "BArray.h"
#include "utils.h"

// --------------------------------- lifecycle ---------------------------------
BArray *BArray_alloc(size_t size) {
    BArray *self = XMALLOC(sizeof(BArray));

    self->size     = size;
    self->n_chunks = PAD(size, 8) / 8;
    self->data     = XMALLOC(self->n_chunks * sizeof(uint8_t));
    self->list     = XMALLOC(size * sizeof(uint16_t));

    BArray_clear(self);

    return self;
}

void BArray_free(BArray *self) {
    free(self->data);
    free(self->list);
    free(self);
}

void BArray_dump(BArray *self) {
    BArray_update(self);
    for (size_t i = 0; i < self->size; ++i)
        printf("%c", BArray_test(self, i) ? '1' : '0');
    printf("\n");
}

// --------------------------------- iterator ----------------------------------
size_t BArray_update(BArray *self) {
    if (self->is_dirty_list) {
        size_t counter = 0;
        size_t base    = 0;
        for (size_t i = 0; i < self->n_chunks; ++i) {
            uint8_t val = self->data[i];
            for (size_t j = 0; j < BArray_tb_count[val]; ++j) {
                self->list[counter] = base + BArray_tb_index[val][j];
                counter++;
            }
            base += 8;
        }

        self->counter          = counter;
        self->is_dirty_counter = false;
        self->is_dirty_list    = false;
    }

    return 0;  // for macro
}

// -------------------------------- operations ---------------------------------
void BArray_clear(BArray *self) {
    memset(self->data, 0, self->n_chunks);

    self->counter          = 0;
    self->is_dirty_counter = false;
    self->is_dirty_list    = false;
}

void BArray_set_all(BArray *self) {
    memset(self->data, -1, self->n_chunks);

    self->counter          = self->size;
    self->is_dirty_counter = false;
    self->is_dirty_list    = true;
}

void BArray_copy(BArray *self, BArray *ba) {
    assert(self->size == ba->size);

    memcpy(self->data, ba->data, self->n_chunks);

    self->counter          = ba->counter;
    self->is_dirty_counter = ba->is_dirty_counter;
    self->is_dirty_list    = true;
}

void BArray_union(BArray *self, BArray *ba) {
    assert((ba == NULL) || (self->size == ba->size));

    if ((ba == NULL) || (BArray_counter(ba) == 0))
        return;

    size_t i = 0;
    for (; i + 7 < self->n_chunks; i += 7)
        *((uint64_t *)(self->data + i)) |= *((uint64_t *)(ba->data + i));
    for (; i < self->n_chunks; ++i)
        self->data[i] |= ba->data[i];

    self->is_dirty_counter = true;
    self->is_dirty_list    = true;
}
