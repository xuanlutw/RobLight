#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

#include "Common.h"
#include "Matrix.h"
#include "PGraph.h"

// ----------------------------------- type ------------------------------------
typedef struct BManager BManager;

// --------------------------------- lifecycle ---------------------------------
BManager *BManager_alloc(Common *common, PGraph *G);
void BManager_free(BManager *self);

// ---------------------------------- getter -----------------------------------
MAT *BManager_lb(BManager *self, size_t l);
MAT *BManager_ub(BManager *self, size_t l);

// -------------------------------- statistics ---------------------------------
void BManager_set_profiling(BManager *self, bool profiling);
clock_t BManager_clock(BManager *self);

// ---------------------------------- status -----------------------------------
void BManager_init(BManager *self);
void BManager_push(BManager *self);
void BManager_pop(BManager *self);
void BManager_flush(BManager *self);

// ------------------------------------ sat ------------------------------------
bool BManager_unsat(BManager *self);
