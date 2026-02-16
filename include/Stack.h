#pragma once

#include "utils.h"
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

// ----------------------------------- node ------------------------------------
typedef struct Node Node;
struct Node {
    Node  *next_node;
    size_t type;
};

static inline Node *Node_push(Node *self, Node *node) {
    node->next_node = self;
    return node;
}

static inline Node *Node_pop(Node *self) {
    return (self == NULL) ? NULL : self->next_node;
}

static inline Node *Node_top(Node *self) {
    return self;
}

static inline void Node_free(Node *self, void (*free_hook)(Node *)) {
    Node *node;
    while ((node = Node_top(self)) != NULL) {
        self = Node_pop(self);
        if (free_hook == NULL)
            free(node);
        else
            free_hook(node);
    }
}

// ----------------------------------- stack -----------------------------------
typedef struct {
    size_t size;
    size_t n_types;

    void (*free_hook)(Node *);

    Node  *nodes;
    Node **free_nodes;
    bool   fresh;
} Stack;

static inline Stack *Stack_alloc(size_t size, size_t n_types,
                                 void (*free_hook)(Node *)) {
    assert(size > sizeof(Node));

    Stack *self      = XMALLOC(sizeof(Stack));
    self->size       = size;
    self->n_types    = n_types;
    self->free_hook  = free_hook;
    self->nodes      = NULL;
    self->free_nodes = XMALLOC((n_types + 1) * sizeof(Node *));
    for (size_t i = 0; i <= n_types; ++i)
        self->free_nodes[i] = NULL;

    return self;
}

static inline void Stack_free(Stack *self) {
    Node_free(self->nodes, self->free_hook);

    for (size_t i = 0; i <= self->n_types; ++i)
        Node_free(self->free_nodes[i], self->free_hook);
    free(self->free_nodes);

    free(self);
}

static inline Node *Stack_push(Stack *self, size_t type) {
    assert(type <= self->n_types);

    Node *node = Node_top(self->free_nodes[type]);

    if (node == NULL) {
        // create new node
        node        = XMALLOC(self->size);
        node->type  = type;
        self->fresh = true;
    }
    else {
        // pop node
        self->free_nodes[type] = Node_pop(self->free_nodes[type]);
        self->fresh            = false;
    }

    self->nodes = Node_push(self->nodes, node);

    return node;
}

static inline void Stack_push_marker(Stack *self) {
    Stack_push(self, 0);
}

static inline bool Stack_fresh(Stack *self) {
    return self->fresh;
}

static inline Node *Stack_pop(Stack *self) {
    assert(self->nodes != NULL);

    Node  *node            = Node_top(self->nodes);
    size_t type            = node->type;
    self->nodes            = Node_pop(self->nodes);
    self->free_nodes[type] = Node_push(self->free_nodes[type], node);

    return (type == 0) ? NULL : node;
}
