cmake_minimum_required(VERSION 3.15)
project(pclib)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Add pybind11
add_subdirectory(extern/pybind11)

# List all source files
set(SOURCES
    src/binding.cpp
)

# Add the pybind11 module
pybind11_add_module(pclib ${SOURCES})

# Include directories
target_include_directories(pclib
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(pclib
    PRIVATE
        Eigen3::Eigen
)

# Set compiler flags for macOS
if(APPLE)
    target_compile_options(pclib PRIVATE 
        -stdlib=libc++ 
        -O3
        -Wall
        -Wextra
        -mmacosx-version-min=10.9
    )
    set_target_properties(pclib PROPERTIES
        LINK_FLAGS "-stdlib=libc++ -mmacosx-version-min=10.9"
    )
endif()
