CUDA_INSTALL_PATH := /usr/local/cuda

CXX := g++
CC := gcc
LINK := g++ -fPIC
NVCC := nvcc

INCLUDES = -I eigen-git-mirror/ -I include/

NVCCFLAGS = -I eigen-git-mirror/ -I include/

CXXFLAGS += $(INCLUDES)

LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart
OBJS = main.cpp.o test.cu.o
TARGET = main
LINKLINE := $(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA)

all: build

build: $(TARGET)

.SUFFIXES: .c .cpp .cu .o
%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS) Makefile
	$(LINKLINE)

clean:
	rm -f *.o $(TARGET)

run: $(TARGET)
	./$(TARGET)
