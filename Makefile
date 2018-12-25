CUDA_INSTALL_PATH := /usr/local/cuda

CXX := g++
CC := gcc
LINK := g++ -dc -fPIC
NVCC := nvcc -rdc=true

INCLUDES = -Ieigen-git-mirror/ -Iinclude/

NVCCFLAGS = -Ieigen-git-mirror/ -Iinclude/

ALL_CCFLAGS += -dc -Xptxas -dlcm=ca

CXXFLAGS += -std=c++17 $(INCLUDES)
LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcurand -lboost_serialization -lcudadevrt
GENCODE_FLAGS := -gencode arch=compute_75,code=sm_75
OBJS = main.cpp.o test.cu.o test_link.cu.o
TARGET = main
LINKLINE := $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA)

all: build

build: $(TARGET)

randGen: randGen.cuh randGen.cu
	$(NVCC) $(NVCCFLAGS) randGen.cu -o randGen $(LIB_CUDA)

main.cpp.o: main.cpp test.h constants.h init.h dataset.h state.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.cpp.o

test.cu.o: test.cu err.cuh cuda_state.cuh state.h test.h randGen.cuh cuda_utility.cuh constants.h input.cuh
	echo "dogs"
	$(NVCC) $(NVCCFLAGS) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c test.cu -o test.cu.o
	nvcc $(GENCODE_FLAGS) -dlink -o test_link.cu.o test.cu.o -lcudart -lcudadevrt

.SUFFIXES: .c .cpp .cu .o
%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS) Makefile
	echo "cats"
	$(LINKLINE)

clean:
	rm -f *.o *.d $(TARGET)

run: $(TARGET)
	./$(TARGET)
