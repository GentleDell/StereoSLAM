# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(STEREO_SLAM)

find_package(OpenCV REQUIRED)
find_package(GSL)   # GNU Scientific Library (for polynomial roots 15/07/2018,zhant.deng)


# *************** OpenCV *****************
message(STATUS "**OpenCV library status:")
message(STATUS "    //version: ${OpenCV_VERSION}")
message(STATUS "    //libraries: ${OpenCV_LIBS}")
message(STATUS "    //include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()


# **************** GSL *******************
set(GSL_LIBS gsl;gslcblas;m)
message(STATUS "**GSL library status:")
message(STATUS "    //version: ${GSL_VERSION}")
message(STATUS "    //libraries: ${GSL_LIBS}")
message(STATUS "    //include path: ${GSL_INCLUDE_DIRS}")


# **************** Mine ******************
include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    )
add_library(${PROJECT_NAME} SHARED
    src/Frame.cpp
    )
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${GSL_LIBS}
    )



# Declare the executable target built from your sources
add_executable(stereo_slam Example/system.cpp)

# Link your application with OpenCV libraries
target_link_libraries(stereo_slam ${OpenCV_LIBS})
target_link_libraries(stereo_slam ${GSL_LIBS})
target_link_libraries(stereo_slam ${PROJECT_NAME})
