#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>

#include "BArray.h"
#include "Common.h"
#include "Stack.h"

typedef struct {
    Common *common;

    size_t  gbudget_max;
    size_t  gbudget;
    size_t *lbudget;

    BArray **lbudget_map;
    BArray  *updated;

    Stack *stack;
} BudManager;

// --------------------------------- lifecycle ---------------------------------
BudManager *BudManager_alloc(Common *common);
void BudManager_free(BudManager *self);
void BudManager_dump(BudManager *self);

// ----------------------------- basic operations ------------------------------
static inline size_t BudManager_gbudget(BudManager *self) {
    return self->gbudget;
}

static inline size_t BudManager_lbudget(BudManager *self, size_t v) {
    if (self->gbudget > self->lbudget[v])
        return self->lbudget[v];
    else
        return self->gbudget;
}

static inline BArray *BudManager_updated(BudManager *self) {
    return self->updated;
}

void BudManager_decrease(BudManager *self, size_t v, size_t u);
size_t BudManager_down(BudManager *self);

// ---------------------------------- status -----------------------------------
void BudManager_init(BudManager *self, size_t gbudget, size_t lbudget);
void BudManager_push(BudManager *self);
void BudManager_pop(BudManager *self);
