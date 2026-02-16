#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "BArray.h"
#include "BudManager.h"
#include "Common.h"
#include "Stack.h"
#include "utils.h"

// ----------------------------------- type ------------------------------------
typedef struct {
    Node node;

    size_t v;
    size_t u;
} BudManagerOp;

// --------------------------------- liftcycle ---------------------------------
BudManager *BudManager_alloc(Common *common) {
    BudManager *self = XMALLOC(sizeof(BudManager));

    self->common  = common;
    self->lbudget = XMALLOC(N_VERTICES * sizeof(size_t));

    self->stack = Stack_alloc(sizeof(BudManagerOp), 1, NULL);

    return self;
}

void BudManager_free(BudManager *self) {
    free(self->lbudget);
    for (size_t i = 0; i <= self->gbudget_max; ++i)
        BArray_free(self->lbudget_map[i]);
    free(self->lbudget_map);

    Stack_free(self->stack);

    free(self);
}

void BudManager_dump(BudManager *self) {
    printf("global budget: %ld\n", BudManager_gbudget(self));
    ITER_VTXS(v) {
        printf("local budget of %ld: %ld\n", v, BudManager_lbudget(self, v));
    }
}

// ----------------------------- basic operations ------------------------------
void BudManager_decrease(BudManager *self, size_t v, size_t u) {
    assert(BudManager_gbudget(self) > 0);

    // decrease budgets
    --self->gbudget;
    if (IS_DIRECTED || (v == u)) {  // self-loop for undirected graph
        BArray_unset(self->lbudget_map[self->lbudget[v]], v);
        --self->lbudget[v];
    }
    if (IS_UNDIRECTED) {
        BArray_unset(self->lbudget_map[self->lbudget[v]], v);
        BArray_unset(self->lbudget_map[self->lbudget[u]], u);
        --self->lbudget[v];
        --self->lbudget[u];
    }

    // push
    BudManagerOp *op = (BudManagerOp *)Stack_push(self->stack, 1);
    op->v            = v;
    op->u            = u;

    // set updated
    self->updated = self->lbudget_map[self->gbudget + 1];
}

size_t BudManager_down(BudManager *self) {
    size_t ret    = self->gbudget;
    self->gbudget = 0;
    ITER_VTXS(w) {
        self->lbudget[w] -= self->gbudget;
    }
    return ret;
}

// ---------------------------------- status -----------------------------------
void BudManager_init(BudManager *self, size_t gbudget, size_t lbudget) {
    lbudget           = (lbudget > gbudget) ? gbudget : lbudget;
    self->gbudget_max = gbudget;
    self->gbudget     = gbudget;
    ITER_VTXS(v) {
        self->lbudget[v] = lbudget;
    }

    self->lbudget_map = XMALLOC((gbudget + 1) * sizeof(BArray *));
    for (size_t i = 0; i <= gbudget; ++i) {
        self->lbudget_map[i] = BArray_alloc(N_VERTICES);
        if (i <= lbudget)
            BArray_set_all(self->lbudget_map[i]);
        else
            BArray_clear(self->lbudget_map[i]);
    }

    self->updated = NULL;
}

void BudManager_push(BudManager *self) {
    Stack_push_marker(self->stack);
    self->updated = NULL;
}

void BudManager_pop(BudManager *self) {
    bool flag = false;

    BudManagerOp *op;
    while ((op = (BudManagerOp *)Stack_pop(self->stack)) != NULL) {
        // restore local budget
        if (IS_DIRECTED) {
            ++self->lbudget[op->v];
            BArray_set(self->lbudget_map[self->lbudget[op->v]], op->v);
        }
        if (IS_UNDIRECTED) {
            ++self->lbudget[op->v];
            ++self->lbudget[op->u];
            BArray_set(self->lbudget_map[self->lbudget[op->v]], op->v);
            BArray_set(self->lbudget_map[self->lbudget[op->u]], op->u);
        }
        flag = true;
    }

    // increase global budget only once
    if (flag)
        self->gbudget++;
}
