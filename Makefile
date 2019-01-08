CUDA_INSTALL_PATH := /usr/local/cuda

CXX := g++
CC := gcc
LINK := g++ -dc -fPIC
NVCC := nvcc

INCLUDES = -Ieigen-git-mirror/ -Iinclude/
NVCCINCLUDES = -Ieigen-git-mirror/ -Iinclude/

NVCCFLAGS = -rdc=true -Xptxas -v

ALL_CCFLAGS += -dc -Xptxas -dlcm=ca

CXXFLAGS += -std=c++17 $(INCLUDES)
LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcurand -lboost_serialization -lcudadevrt
GENCODE_FLAGS := -gencode arch=compute_75,code=sm_75

OBJDIR = obj
BINDIR = bin

TARGET = $(BINDIR)/main

SOURCES := $(wildcard **/*.cpp)
OBJECTS := $(patsubst %.cpp,$(OBJDIR)/%.o,$(SOURCES))
DEPS = $(OBJECTS:%.o=%.d)

CUDA_SOURCES := $(wildcard **/*.cu)
CUDA_OBJECTS := $(patsubst %.cu,$(OBJDIR)/%.cu.o,$(CUDA_SOURCES))
CUDA_LINK_OBJECTS := $(patsubst %.cu,$(OBJDIR)/%_link.cu.o,$(CUDA_SOURCES))
CUDA_DEPS = $(CUDA_OBJECTS:%.cu.o=%.cu.d)

all: build

build: $(TARGET)

run: $(TARGET)
	./$(TARGET)

-include $(DEPS)
-include $(CUDA_DEPS)

$(CUDA_DEPS): $(OBJDIR)/%.cu.d : %.cu
	@mkdir -p $(dir $@)
	@echo "Building deps: $@"
	@ set -e; rm -f $@; \
	$(NVCC) -M -MT $@ $(NVCCFLAGS) $(NVCCINCLUDES) $< > $@.$$$$; \
	sed 's,\($*\)\.cu.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

$(OBJECTS): $(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

$(CUDA_OBJECTS): $(OBJDIR)/%.cu.o : %.cu
	@mkdir -p $(dir $@)
	@$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $< -o $@
	@nvcc $(GENCODE_FLAGS) -dlink -o $(OBJDIR)/$*_link.cu.o $@ -lcudart -lcudadevrt

$(TARGET): $(OBJECTS) $(CUDA_OBJECTS) Makefile
	@mkdir -p $(dir $@)
	$(LINK) -o $(TARGET) $(OBJECTS) $(CUDA_OBJECTS) $(CUDA_LINK_OBJECTS) $(LIB_CUDA)

.PHONY : clean
clean :
	@rm -rf $(OBJDIR) $(BINDIR)
