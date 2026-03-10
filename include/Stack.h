#pragma once

#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "utils.h"

// ----------------------------------- node ------------------------------------
typedef struct Node Node;
struct Node {
    Node *next_node;
    bool  is_marker;
};

static inline Node *Node_push(Node *self, Node *node) {
    node->next_node = self;
    return node;
}

static inline Node *Node_pop(Node *self) {
    return (self == NULL) ? NULL : self->next_node;
}

// ----------------------------------- stack -----------------------------------
typedef struct {
    size_t size;

    Node *nodes;
    Node *free_nodes;
    Node *free_markers;

    bool is_fresh;
} Stack;

static inline void Stack_init(Stack *self, size_t size) {
    assert(size > sizeof(Node));

    self->size         = size;
    self->nodes        = NULL;
    self->free_nodes   = NULL;
    self->free_markers = NULL;
}

static inline void Stack_free_nodes(Stack *self __attribute__((unused)),
                                    Node  *node, void (*free_hook)(Node *)) {
    Node *node_top;
    while ((node_top = node) != NULL) {
        node = Node_pop(node_top);
        if ((node_top->is_marker) || (free_hook == NULL))
            free(node_top);
        else
            free_hook(node_top);
    }
}

static inline void Stack_cleanup(Stack *self, void (*free_hook)(Node *)) {
    Stack_free_nodes(self, self->nodes, free_hook);
    Stack_free_nodes(self, self->free_nodes, free_hook);
    Stack_free_nodes(self, self->free_markers, free_hook);
}

static inline Node *Stack_push(Stack *self) {
    Node *node;
    if (self->free_nodes == NULL) {
        // create new node
        node            = XMALLOC(self->size);
        node->is_marker = false;
        self->is_fresh  = true;
    }
    else {
        // pop node
        node             = self->free_nodes;
        self->free_nodes = Node_pop(self->free_nodes);
        self->is_fresh   = false;
    }

    self->nodes = Node_push(self->nodes, node);

    return node;
}

static inline void Stack_push_marker(Stack *self) {
    Node *node;
    if (self->free_markers == NULL) {
        // create new node
        node            = XMALLOC(sizeof(Node));
        node->is_marker = true;
    }
    else {
        // pop node
        node               = self->free_markers;
        self->free_markers = Node_pop(self->free_markers);
    }

    self->nodes = Node_push(self->nodes, node);
}

static inline bool Stack_fresh(Stack *self) {
    return self->is_fresh;
}

static inline Node *Stack_pop(Stack *self) {
    Node *node = self->nodes;
    if (node == NULL) {
        return NULL;
    }
    else if (node->is_marker) {
        self->nodes        = Node_pop(self->nodes);
        self->free_markers = Node_push(self->free_markers, node);
        return NULL;
    }
    else {
        self->nodes      = Node_pop(self->nodes);
        self->free_nodes = Node_push(self->free_nodes, node);
        return node;
    }
}
