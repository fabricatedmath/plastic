CUDA_INSTALL_PATH := /usr/local/cuda

CXX := g++
CC := gcc
LINK := g++ -fPIC
NVCC := nvcc

INCLUDES = -Ieigen-git-mirror/ -Iinclude/

NVCCFLAGS = -Ieigen-git-mirror/ -Iinclude/

CXXFLAGS += -std=c++17 $(INCLUDES)
LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcurand -lboost_serialization
OBJS = main.cpp.o test.cu.o
TARGET = main
LINKLINE := $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA)

all: build

build: $(TARGET)

randGen: randGen.cuh randGen.cu
	$(NVCC) $(NVCCFLAGS) randGen.cu -o randGen $(LIB_CUDA)

main.cpp.o: main.cpp test.h constants.h init.h dataset.h state.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.cpp.o

test.cu.o: test.cu err.cuh cuda_state.cuh state.h test.h randGen.cuh cuda_utility.cuh
	$(NVCC) $(NVCCFLAGS) -c test.cu -o test.cu.o

.SUFFIXES: .c .cpp .cu .o
%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS) Makefile
	$(LINKLINE)

clean:
	rm -f *.o *.d $(TARGET)

run: $(TARGET)
	./$(TARGET)
