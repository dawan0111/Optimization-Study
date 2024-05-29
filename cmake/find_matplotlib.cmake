message(STATUS "Start Finding matplotlib.")

find_package(matplotlib_cpp)

if(matplotlib_cpp_FOUND)
message(STATUS "Found: matplotlib - ${matplotlib_cpp_INCLUDE_DIRS}")
# message(STATUS "Found: matplotlib_LIBS - ${matplotlib_cpp_LIB}")

include_directories(${matplotlib_cpp_INCLUDE_DIRS})
endif()

message(STATUS "Finish Finding matplotlib.\n")