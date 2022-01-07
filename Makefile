NVCC = nvcc
NVCC_FLAGS = -std=c++14 -O3
CXX = g++
CXX_FLAGS = -std=c++14 -O3
LD = -lcublas -ljsoncpp

INC_DIR = include
SRC_DIR = src
OBJ_DIR = bin
$(shell if [ ! -e $(OBJ_DIR) ];then mkdir -p $(OBJ_DIR); fi)

EXE = runme

HEADERS = $(wildcard $(INC_DIR)/*.h)
SRCS = $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(patsubst %.cu, $(OBJ_DIR)/%.o, $(notdir $(SRCS))))

$(EXE): $(OBJS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(LD) $(OBJS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(HEADERS)
	$(CXX) -c $(CXX_FLAGS) -I$(INC_DIR) $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(HEADERS)
	$(NVCC) -c $(NVCC_FLAGS) -I$(INC_DIR) $< -o $@

.PHONY:
clean:
	rm -r $(EXE) $(OBJ_DIR)/*