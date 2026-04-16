cmake_minimum_required(VERSION 3.16)
project(SWIGX_GA_TDOA_3D_polyscope LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Expected layout:
#   <workspace>/polyscope/         (Polyscope source tree)
#   <workspace>/SWIGX_GA_TDOA_3D_polyscope.cpp
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/polyscope/CMakeLists.txt")
  message(FATAL_ERROR
    "Polyscope source not found at ${CMAKE_CURRENT_SOURCE_DIR}/polyscope.\n"
    "Add it first, e.g. git clone https://github.com/nmwsharp/polyscope.git polyscope")
endif()

add_subdirectory(polyscope)

add_executable(SWIGX_GA_TDOA_3D_polyscope
  SWIGX_GA_TDOA_3D_polyscope_v2_improved.cpp
)

target_link_libraries(SWIGX_GA_TDOA_3D_polyscope PRIVATE polyscope)

if(MSVC)
  target_compile_options(SWIGX_GA_TDOA_3D_polyscope PRIVATE /W4)
else()
  target_compile_options(SWIGX_GA_TDOA_3D_polyscope PRIVATE -Wall -Wextra -Wpedantic)
endif()
