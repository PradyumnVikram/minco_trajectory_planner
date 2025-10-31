cmake_minimum_required(VERSION 3.10)

# Set your project name
project(MINCO_Imp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Eigen package
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen: ${EIGEN3_INCLUDE_DIR}")

# Add your source files
add_executable(my_program
    main.cpp
    # Add other .cpp files here
)

# Link Eigen to your executable
target_include_directories(my_program PRIVATE ${EIGEN3_INCLUDE_DIR})
