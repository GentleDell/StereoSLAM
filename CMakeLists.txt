# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(stereo_matching_project)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)
find_package(GSL)   # GNU Scientific Library (for polynomial roots 15/07/2018,zhant.deng)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "**OpenCV library status:")
message(STATUS "    //version: ${OpenCV_VERSION}")
message(STATUS "    //libraries: ${OpenCV_LIBS}")
message(STATUS "    //include path: ${OpenCV_INCLUDE_DIRS}")

set(GSL_LIBS gsl;gslcblas;m)
message(STATUS "**GSL library status:")
message(STATUS "    //version: ${GSL_VERSION}")
message(STATUS "    //libraries: ${GSL_LIBS}")
message(STATUS "    //include path: ${GSL_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Declare the executable target built from your sources
add_executable(stereo_matching matching.cpp)

# Link your application with OpenCV libraries
target_link_libraries(stereo_matching ${OpenCV_LIBS})

target_link_libraries(stereo_matching ${GSL_LIBS})
