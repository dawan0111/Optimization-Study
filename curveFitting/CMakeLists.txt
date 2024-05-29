cmake_minimum_required(VERSION 3.21)

set(PACKAGE_NAME CurveFitting)
set(PACKAGE_VERSION 0.0.1)

project(${PACKAGE_NAME} VERSION ${PACKAGE_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_SOURCE_DIR}/cmake/find_matplotlib.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/find_eigen.cmake)

add_executable(GAUSS_NEWTON gaussNewton.cpp)

target_link_libraries(GAUSS_NEWTON PRIVATE
    matplotlib_cpp::matplotlib_cpp
    ${EIGEN3_LIBS}
)