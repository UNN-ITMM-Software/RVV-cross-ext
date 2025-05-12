cmake_minimum_required(VERSION 3.0.0)

project(pica)


set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(USE_TESTS "Enable unit tests" ON)
option(USE_OPENMP "Use OpenMP" ON)
option(USE_MIC "Use MIC" OFF)
option(USE_AVX "Use AVX" OFF)


set(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
#set(ARCH "x64")


set(LIB_DIR ${ROOT_DIR}/lib)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH ${LIB_DIR})

set(DEPS
)

if (UNIX AND NOT APPLE)
    list(APPEND DEPS rt)
endif()

if (USE_OPENMP)
    find_package(OpenMP)
    if (OPENMP_FOUND)
        message(STATUS "Using OpenMP")
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    else()
        message(WARNING "OpenMP not found")
    endif()
endif()


# In order for pica files to be shown in IDE, create two targets for it
include_directories(include)
add_library(pica_interface INTERFACE)
set(PICA_INCLUDE_DIR include)	
set(PICA_HEADER_DIR ${PICA_INCLUDE_DIR}/pica)
set(pica_headers
    ${PICA_HEADER_DIR}/currentDeposition/CurrentDepositor.h
    ${PICA_HEADER_DIR}/currentDeposition/CurrentDepositorInterface.h
    ${PICA_HEADER_DIR}/currentDeposition/CurrentDepositorYeeGrid.h	
    ${PICA_HEADER_DIR}/fieldInterpolation/FieldInterpolator.h
    ${PICA_HEADER_DIR}/fieldInterpolation/FieldInterpolatorInterface.h
    ${PICA_HEADER_DIR}/fieldInterpolation/FieldInterpolatorYeeGrid.h	
    ${PICA_HEADER_DIR}/fieldSolver/YeeSolver.h
    ${PICA_HEADER_DIR}/grid/Grid.h
    ${PICA_HEADER_DIR}/grid/YeeGrid.h
    ${PICA_HEADER_DIR}/math/Constants.h
    ${PICA_HEADER_DIR}/math/Dimension.h
    ${PICA_HEADER_DIR}/math/FP.h
    ${PICA_HEADER_DIR}/math/Vectors.h
    ${PICA_HEADER_DIR}/particles/Ensemble.h
    ${PICA_HEADER_DIR}/particles/EnsembleOrdered.h
    ${PICA_HEADER_DIR}/particles/EnsembleSupercells.h
    ${PICA_HEADER_DIR}/particles/EnsembleUnordered.h
    ${PICA_HEADER_DIR}/particles/Particle.h
    ${PICA_HEADER_DIR}/particles/ParticleArray.h
    ${PICA_HEADER_DIR}/particles/ParticleInterface.h
    ${PICA_HEADER_DIR}/particles/ParticleTraits.h
    ${PICA_HEADER_DIR}/particlePush/BorisPusher.h
	${PICA_HEADER_DIR}/threading/OpenMPHelper.h
	${PICA_HEADER_DIR}/utility/Array.h
	${PICA_HEADER_DIR}/utility/Assert.h
	${PICA_HEADER_DIR}/utility/Utility.h
    ${PICA_HEADER_DIR}/Parameters.h)
add_custom_target(pica SOURCES ${pica_headers})
