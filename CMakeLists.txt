cmake_minimum_required(VERSION 3.17)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

project(ScientificDB CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optional Integrations (OFF by default)
option(ENABLE_GPU "Enable CUDA/GPU support" OFF)
option(ENABLE_MYSQL "Enable MariaDB database integration" OFF)
set(MARIADB_DIR "$ENV{HOME}/Projects/CPPProj/MariaDB" CACHE PATH "Path to MariaDB source/build root")

include(FetchContent)

# 1. Fetch Chemfiles
FetchContent_Declare(
    chemfiles
    GIT_REPOSITORY https://github.com/chemfiles/chemfiles.git
    GIT_TAG 0.10.4
)
FetchContent_MakeAvailable(chemfiles)

# 2. Fetch Polyscope
FetchContent_Declare(
    polyscope
    GIT_REPOSITORY https://github.com/nmwsharp/polyscope.git
    GIT_TAG master
)
FetchContent_MakeAvailable(polyscope)

# Header Patches for modern GCC (kept safe if older headers are fetched)
if(EXISTS ${polyscope_SOURCE_DIR}/include/polyscope/internal.h)
    file(READ ${polyscope_SOURCE_DIR}/include/polyscope/internal.h POLYSCOPE_INTERNAL_H)
    if(NOT "${POLYSCOPE_INTERNAL_H}" MATCHES "#include <cstdint>")
        string(REPLACE "#include <string>" "#include <string>\n#include <cstdint>" POLYSCOPE_INTERNAL_H "${POLYSCOPE_INTERNAL_H}")
        file(WRITE ${polyscope_SOURCE_DIR}/include/polyscope/internal.h "${POLYSCOPE_INTERNAL_H}")
    endif()
endif()

if(EXISTS ${polyscope_SOURCE_DIR}/include/polyscope/weak_handle.h)
    file(READ ${polyscope_SOURCE_DIR}/include/polyscope/weak_handle.h POLYSCOPE_WEAK_HANDLE_H)
    if(NOT "${POLYSCOPE_WEAK_HANDLE_H}" MATCHES "#include <cstdint>")
        string(REPLACE "#include <stdexcept>" "#include <stdexcept>\n#include <cstdint>" POLYSCOPE_WEAK_HANDLE_H "${POLYSCOPE_WEAK_HANDLE_H}")
        file(WRITE ${polyscope_SOURCE_DIR}/include/polyscope/weak_handle.h "${POLYSCOPE_WEAK_HANDLE_H}")
    endif()
endif()

# Find OpenMP and TBB for parallel execution
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)

# Base include and link paths (CPU)
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    /usr/local/include
)

link_directories(
    /usr/local/lib64
)

add_executable(ScientificDB main.cpp)

# Link base CPU dependencies
target_link_libraries(ScientificDB PRIVATE 
    chemfiles 
    polyscope
    readline
    faiss
    openblas
    OpenMP::OpenMP_CXX
    TBB::tbb
)

# Configure CUDA/GPU if enabled
if(ENABLE_GPU)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(ScientificDB PRIVATE USE_GPU=1)
    target_link_libraries(ScientificDB PRIVATE 
        CUDA::cudart
        CUDA::cublas
    )
    message(STATUS "[ScientificDB] GPU/CUDA support ENABLED.")
else()
    target_compile_definitions(ScientificDB PRIVATE USE_GPU=0)
    message(STATUS "[ScientificDB] GPU/CUDA support DISABLED (CPU-only mode).")
endif()

# Configure MariaDB if enabled
if(ENABLE_MYSQL)
    # Locate libmariadb across the actual nested source-build output paths
    find_library(MARIADB_CLIENT_LIB
        NAMES mariadb mariadbclient mysqlclient
        PATHS 
            "${MARIADB_DIR}/build/libmariadb/libmariadb"
            "${MARIADB_DIR}/build/libmariadb"
            "${MARIADB_DIR}/lib"
            "${MARIADB_DIR}/lib64"
        PATH_SUFFIXES Release Debug
    )
    
    # Fallback to system-wide search if still not found
    if(NOT MARIADB_CLIENT_LIB)
        find_library(MARIADB_CLIENT_LIB NAMES mariadb mariadbclient mysqlclient REQUIRED)
    endif()

    target_compile_definitions(ScientificDB PRIVATE USE_MYSQL=1)
    target_include_directories(ScientificDB PRIVATE
        "${MARIADB_DIR}/libmariadb/include"
        "${MARIADB_DIR}/build/libmariadb/include"
        "${MARIADB_DIR}/include"
        "${MARIADB_DIR}/build/include"
        "${MARIADB_DIR}/libmariadb"
    )
    target_link_libraries(ScientificDB PRIVATE ${MARIADB_CLIENT_LIB})

    # Embed RPATH so the binary finds libmariadb.so at runtime without LD_LIBRARY_PATH
    get_filename_component(MARIADB_LIB_DIR "${MARIADB_CLIENT_LIB}" DIRECTORY)
    set_target_properties(ScientificDB PROPERTIES
        BUILD_RPATH "${MARIADB_LIB_DIR}"
        INSTALL_RPATH "${MARIADB_LIB_DIR}"
    )

    message(STATUS "[ScientificDB] MariaDB support ENABLED using: ${MARIADB_CLIENT_LIB}")
else()
    target_compile_definitions(ScientificDB PRIVATE USE_MYSQL=0)
    message(STATUS "[ScientificDB] MariaDB support DISABLED (offline mode).")
endif()