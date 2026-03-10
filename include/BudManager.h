#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "Stack.h"
#include "utils.h"

typedef struct {
    size_t gbudget_max;
    size_t lbudget_max;

    uint16_t  gbudget;
    uint16_t *lbudget;

    Stack stack;
} BudManager;

// --------------------------------- lifecycle ---------------------------------
void BudManager_init(BudManager *self, bool comp_radius, uint16_t gbudget,
                     uint16_t lbudget);

static inline void BudManager_init_v(BudManager *self, size_t v, size_t deg) {
    self->lbudget[v] = MIN(self->lbudget_max, deg);
}

void BudManager_cleanup(BudManager *self);

// ----------------------------- basic operations ------------------------------
static inline uint16_t BudManager_gbudget(BudManager *self) {
    return self->gbudget;
}

static inline uint16_t BudManager_lbudget(BudManager *self, size_t v) {
    if (self->gbudget > self->lbudget[v])
        return self->lbudget[v];
    else
        return self->gbudget;
}

void BudManager_decrease(BudManager *self, size_t v, size_t u);
void BudManager_down(BudManager *self, uint16_t delta);
uint16_t *BudManager_update(BudManager *self, bool *flag, uint16_t *queue);
// ---------------------------------- status -----------------------------------
void BudManager_push(BudManager *self);
void BudManager_pop(BudManager *self);
