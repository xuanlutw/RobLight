#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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
void BudManager_init(BudManager *self, bool comp_radius, uint16_t gbudget,
                     uint16_t lbudget) {
    self->lbudget = XMALLOC(N_VERTICES * sizeof(uint16_t));

    Stack_init(&self->stack, sizeof(BudManagerOp));

    self->gbudget_max = comp_radius ? N_EDGES : gbudget;
    self->lbudget_max = comp_radius ? N_EDGES : MIN(gbudget, lbudget);

    self->gbudget = self->gbudget_max;
}

void BudManager_cleanup(BudManager *self) {
    free(self->lbudget);

    Stack_cleanup(&self->stack, NULL);
}

// ----------------------------- basic operations ------------------------------
void BudManager_decrease(BudManager *self, size_t v, size_t u) {
    assert(BudManager_gbudget(self) > 0);

    // decrease budgets
    --self->gbudget;
    if (IS_DIRECTED || (v == u)) {  // self-loop for undirected graph
        --self->lbudget[v];
    }
    else {
        assert(IS_UNDIRECTED && (v != u));
        --self->lbudget[v];
        --self->lbudget[u];
    }

    // push
    BudManagerOp *op = (BudManagerOp *)Stack_push(&self->stack);
    op->v            = v;
    op->u            = u;
}

void BudManager_down(BudManager *self, uint16_t delta) {
    assert(delta <= self->gbudget);

    self->gbudget -= delta;
    ITER_VTXS(w) {
        self->lbudget[w] = MIN(self->gbudget, self->lbudget[w]);
    }
}

uint16_t *BudManager_update(BudManager *self, bool *flag, uint16_t *queue) {
    ITER_VTXS(v) {
        if (!flag[v] && (self->lbudget[v] > self->gbudget)) {
            flag[v] = true;
            *queue  = v;
            ++queue;
        }
    }

    return queue;
}

// ---------------------------------- status -----------------------------------
void BudManager_push(BudManager *self) {
    Stack_push_marker(&self->stack);
}

void BudManager_pop(BudManager *self) {
    bool flag = false;

    BudManagerOp *op;
    while ((op = (BudManagerOp *)Stack_pop(&self->stack)) != NULL) {
        // restore local budget
        if (IS_DIRECTED) {
            ++self->lbudget[op->v];
        }
        else {
            assert(IS_UNDIRECTED);
            ++self->lbudget[op->v];
            ++self->lbudget[op->u];
        }
        flag = true;
    }

    // increase global budget only once
    if (flag)
        self->gbudget++;
}
