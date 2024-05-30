message(STATUS "Start Finding Ceres.")

find_package(Ceres)

if(Ceres_FOUND)
message(STATUS "Found: Eigen3 - ${CERES_INCLUDE_DIRS}")
include_directories(${CERES_INCLUDE_DIRS})
endif()

message(STATUS "Finish Finding Ceres.\n")