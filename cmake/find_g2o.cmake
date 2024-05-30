message(STATUS "Start Finding g2o.")

find_package(g2o)

if(g2o_FOUND)
message(STATUS "Found: g2o - ${G2O_INCLUDE_DIR}")
include_directories(${G2O_INCLUDE_DIR})
endif()

message(STATUS "Finish Finding g2o.\n")