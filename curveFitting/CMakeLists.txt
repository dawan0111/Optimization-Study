cmake_minimum_required(VERSION 3.21)

set(PACKAGE_NAME CurveFitting)
set(PACKAGE_VERSION 0.0.1)

project(${PACKAGE_NAME} VERSION ${PACKAGE_VERSION} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_SOURCE_DIR}/cmake/find_matplotlib.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/find_eigen.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/find_ceres.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/find_g2o.cmake)

add_executable(GAUSS_NEWTON gaussNewton.cpp)

target_link_libraries(GAUSS_NEWTON PRIVATE
    matplotlib_cpp::matplotlib_cpp
    ${EIGEN3_LIBS}
)

add_executable(CERES ceres.cpp)

target_link_libraries(CERES PRIVATE
    matplotlib_cpp::matplotlib_cpp
    ${EIGEN3_LIBS}
    ${CERES_LIBRARIES}
)

add_executable(G2O g2o.cpp)

target_link_libraries(G2O PRIVATE
    matplotlib_cpp::matplotlib_cpp
    ${EIGEN3_LIBS}
    ${CERES_LIBRARIES}
    g2o_core
    g2o_stuff
)