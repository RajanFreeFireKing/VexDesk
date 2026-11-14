cmake_minimum_required(VERSION 3.16)
project(StreamVex VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -fPIC")
endif()

if(ANDROID)
    if(${ANDROID_ABI} STREQUAL "arm64-v8a")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+dotprod")
    endif()
endif()

set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build server" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    add_subdirectory(third_party/llama.cpp)
    set(LLAMA_LIBS llama ggml)
    set(LLAMA_INCLUDES 
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include 
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    )
else()
    set(LLAMA_LIBS "")
    set(LLAMA_INCLUDES "")
endif()

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LLAMA_INCLUDES}
)

find_package(Threads REQUIRED)

add_library(streamvex SHARED
    src/streamvex.cpp
    src/jni_bridge.cpp
    src/io/layer_streamer.cpp
    src/parser/gguf_reader.cpp
    src/compute/dequant.cpp
    src/compute/gemv.cpp
    src/compute/transformer_layer.cpp
    src/compute/sampler.cpp
)

target_link_libraries(streamvex PRIVATE ${LLAMA_LIBS} Threads::Threads)

target_include_directories(streamvex PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_definitions(streamvex PRIVATE STREAMVEX_BUILDING_DLL)

set_target_properties(streamvex PROPERTIES
    OUTPUT_NAME "streamvex"
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
)
