cmake_minimum_required(VERSION 3.13.4)

project(Plastic LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -rdc=true --expt-relaxed-constexpr -gencode arch=compute_75,code=sm_75")

include_directories(${Plastic_SOURCE_DIR}/eigen-cuda-utility/include)

file(GLOB gpu_source_files "${Plastic_SOURCE_DIR}/src/*.cu")

#use find eigen?
# find_package (Eigen3 3.3 REQUIRED NO_MODULE)
# add_executable (example example.cpp)
# target_link_libraries (example Eigen3::Eigen)
include_directories(${Plastic_SOURCE_DIR}/eigen-git-mirror)

include_directories(${Plastic_SOURCE_DIR}/src)
include_directories(${Plastic_SOURCE_DIR}/src/kernels/robust_kernel)

find_package(Boost COMPONENTS serialization program_options REQUIRED)

add_executable(plastic ${Plastic_SOURCE_DIR}/src/main.cpp ${gpu_source_files})

target_link_libraries(plastic ${Boost_LIBRARIES})
target_link_libraries(plastic curand)
