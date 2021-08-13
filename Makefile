NVCC = nvcc
NVCC_FLAGS = -std=c++14 -g -O3
LD = -lcublas -ljsoncpp

OBJ_DIR = bin

HEADERS = $(wildcard *.h)
SRCS = $(wildcard *.cu)
OBJS = $(patsubst %.cu, $(OBJ_DIR)/%.o, $(SRCS))

runme: $(OBJS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(LD) $(OBJS) -o $@

$(OBJ_DIR)/%.o: %.cu $(HEADERS)
	$(NVCC) -c $(NVCC_FLAGS) $< -o $@

.PHONY:
clean:
	rm -r runme $(OBJ_DIR)/**
