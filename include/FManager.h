#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <time.h>

#include "Common.h"
#include "Matrix.h"
#include "PGraph.h"

// ----------------------------------- type ------------------------------------
typedef struct FManager FManager;

// --------------------------------- lifecycle ---------------------------------
FManager *FManager_alloc(Common *common, PGraph *G);
void FManager_free(FManager *self);

// ---------------------------------- getter -----------------------------------
MAT *FManager_feat(FManager *self, size_t l);

// -------------------------------- statistics ---------------------------------
void FManager_set_profiling(FManager *self, bool profiling);
clock_t FManager_clock(FManager *self);

// ---------------------------------- status -----------------------------------
void FManager_init(FManager *self);
void FManager_push(FManager *self);
void FManager_pop(FManager *self);

// ------------------------------------ sat ------------------------------------
bool FManager_sat(FManager *self);
