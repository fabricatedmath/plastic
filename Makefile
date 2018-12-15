CXX := g++
CC := gcc
LINK := g++ -fPIC
NVCC := nvcc

INCLUDES = -I eigen-git-mirror/ -I include/

CXXFLAGS += $(INCLUDES)

OBJS = main.cpp.o
TARGET = main
LINKLINE := $(LINK) -o $(TARGET) $(OBJS)

all: build

build: $(TARGET)

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(LINKLINE)

clean:
	rm -f *.o $(TARGET)

run: $(TARGET)
	./$(TARGET)
