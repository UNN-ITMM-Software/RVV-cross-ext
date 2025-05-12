cmake_minimum_required(VERSION 3.0.0)

project(pica-benchmark)

#    set(CMAKE_CXX_FLAGS "-g -DNDEBUG -O2 -m64 -fopenmp -fpermissive")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
#set(ARCH "x64")

set(LIB_DIR ${ROOT_DIR}/lib)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH ${LIB_DIR})

add_subdirectory(pica)
add_subdirectory(src)