CUDA_INSTALL_PATH := /usr/local/cuda

CXX := g++
CC := gcc
LINK := g++ -dc -fPIC
NVCC := nvcc

INCLUDES = -Ieigen-git-mirror/ -Iinclude/
NVCCINCLUDES = -Ieigen-git-mirror/ -Iinclude/

DEFINES =
#DEFINES = -DNBE=2 -DNBI=1 -DNBSTEPSPERPRESRUN=350 -DNUMPRESTHISLAUNCH=1 -DNUMLOOPS=1

NVCCFLAGS = -lineinfo -rdc=true -Xptxas -v --expt-relaxed-constexpr

ALL_CCFLAGS += -dc #-Xptxas -dlcm=cg

CXXFLAGS += -std=c++17 $(INCLUDES)
LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib64 -lcudart -lcurand -lboost_serialization -lcudadevrt -lboost_program_options
#GENCODE_FLAGS := -gencode arch=compute_61,code=sm_61 -gencode arch=compute_75,code=sm_75
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
	./$(TARGET) ${ARGS}

-include $(DEPS)
-include $(CUDA_DEPS)

$(DEPS): $(OBJDIR)/%.d : %.cpp Makefile
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -MM -MT $(OBJDIR)/$*.o $< -MF $@

$(CUDA_DEPS): $(OBJDIR)/%.cu.d : %.cu Makefile
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -M -MT $(OBJDIR)/$*.cu.o $< > $@

$(OBJECTS): $(OBJDIR)/%.o : %.cpp $(OBJDIR)/%.d Makefile
	$(CXX) $(CXXFLAGS) $(DEFINES) -c $< -o $@

$(CUDA_OBJECTS): $(OBJDIR)/%.cu.o : %.cu $(OBJDIR)/%.cu.d Makefile
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(DEFINES) $(NVCCINCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $< -o $@
	$(NVCC) $(GENCODE_FLAGS) -dlink -o $(OBJDIR)/$*_link.cu.o $@ -lcudart -lcudadevrt

$(TARGET): $(OBJECTS) $(CUDA_OBJECTS) Makefile
	@mkdir -p $(dir $@)
	$(LINK) -o $(TARGET) $(OBJECTS) $(CUDA_OBJECTS) $(CUDA_LINK_OBJECTS) $(LIB_CUDA)

.PHONY : clean
clean :
	@rm -rf $(OBJDIR) $(BINDIR)
