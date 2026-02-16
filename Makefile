# Compiler and flags
CC     = gcc
CFLAGS = -std=c11 -I${INC_DIR} -Wall -Wextra -march=native -flto -MMD -MP
LFLAGS = -flto

CFLAGS_RELEASE = $(CFLAGS) -O3 -DNDEBUG

# Folders
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj

OBJ_RELEASE_DIR = $(OBJ_DIR)/release

# Files
SRCS = $(wildcard $(SRC_DIR)/*.c)

OBJS_RELEASE = $(patsubst $(SRC_DIR)/%.c, $(OBJ_RELEASE_DIR)/%.o, $(SRCS))

# Dependency
DEPS_RELEASE = $(OBJS_RELEASE:.o=.d)

# Targets
BIN_RELEASE = roblight

# Rules
.PHONY: all release clean

all: release

$(OBJ_RELEASE_DIR):
	mkdir -p $@

release: $(BIN_RELEASE)

# Link
$(BIN_RELEASE): $(OBJS_RELEASE)
	$(CC) $(LFLAGS) -o $@ $^

# Compile
$(OBJ_RELEASE_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_RELEASE_DIR)
	$(CC) $(CFLAGS_RELEASE) -c $< -o $@

-include $(DEPS_RELEASE)

# Clean
clean:
	rm -rf $(OBJ_DIR) $(BIN_RELEASE)
