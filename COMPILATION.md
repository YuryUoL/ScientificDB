# Building ScientificDB

`ScientificDB` is a C++17 application that uses **CMake** to manage its build and dependencies. It automatically fetches **Chemfiles** and **Polyscope** using CMake's `FetchContent`, while linking against system-installed **FAISS**, **OpenBLAS**, **Intel oneTBB**, **OpenMP**, and **GNU Readline**.

Both **CUDA/GPU support** and **MariaDB/MySQL integration** are optional and disabled by default:

- `ENABLE_GPU=OFF` — builds in CPU-only mode without requiring the CUDA Toolkit.
- `ENABLE_MYSQL=OFF` — disables database integration, avoiding the need for MariaDB/MySQL client libraries.

This allows ScientificDB to compile and run in standalone CPU-only mode without CUDA or a database installation.

---

## 1. System Prerequisites

Before compiling, ensure you have:

- **C++17-compatible compiler** (`g++` or `clang++`)
- **CMake 3.17+**
- **Git**
- **FAISS** (headers in `/usr/local/include`, libraries in `/usr/local/lib64`)
- **OpenBLAS**
- **Intel oneTBB** (`tbb`)
- **OpenMP**
- **GNU Readline** (`readline`)

The following dependencies are fetched automatically by CMake and do not need to be installed manually:

- **Chemfiles** (version 0.10.4)
- **Polyscope** (master branch)

### Example Package Installation (Fedora / RHEL)

```bash
sudo dnf install cmake gcc-c++ git tbb-devel openblas-devel readline-devel
```

You must also have FAISS installed and accessible through the configured include and library paths.

---

## 2. Standard CPU-Only Build (Default)

The default configuration requires neither CUDA nor MariaDB.

```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

Run the application:

```bash
./build/ScientificDB
```

---

## 3. Optional CUDA/GPU Support

To enable GPU acceleration, install the CUDA Toolkit and configure the project with:

```bash
cmake -S . -B build -DENABLE_GPU=ON
cmake --build build -j$(nproc)
```

CMake will locate the CUDA Toolkit and link against:

- CUDA Runtime (`cudart`)
- CUDA BLAS (`cublas`)

When enabled, the application is compiled with `USE_GPU=1`. Otherwise, it uses `USE_GPU=0`.

---

## 4. Optional MariaDB/MySQL Integration

To enable MariaDB/MySQL database integration, configure the project with:

```bash
cmake -S . -B build -DENABLE_MYSQL=ON
cmake --build build -j$(nproc)
```

By default, CMake searches for MariaDB client libraries in:

```text
~/Projects/CPPProj/MariaDB
```

You can override this location using `MARIADB_DIR`:

```bash
cmake -S . -B build \
    -DENABLE_MYSQL=ON \
    -DMARIADB_DIR=/path/to/MariaDB
```

CMake searches for the MariaDB client library (`mariadb`, `mariadbclient`, or `mysqlclient`) and configures the appropriate include paths and runtime library path (RPATH).

When enabled, the application is compiled with `USE_MYSQL=1`.

---

## 5. Enable Both CUDA and MariaDB

Both integrations can also be enabled simultaneously:

```bash
cmake -S . -B build \
    -DENABLE_GPU=ON \
    -DENABLE_MYSQL=ON

cmake --build build -j$(nproc)
```

---

## 6. Build Configuration Summary

| Feature | CMake option | Default |
|---|---|---|
| GPU / CUDA support | `ENABLE_GPU` | OFF |
| MariaDB / MySQL integration | `ENABLE_MYSQL` | OFF |
| C++ standard | `CMAKE_CXX_STANDARD` | C++17 |

**Note:** Chemfiles and Polyscope are downloaded automatically during CMake configuration, so the initial configuration requires an internet connection. Subsequent builds can reuse the downloaded dependencies.

**CPU-only dependencies:** FAISS, OpenBLAS, Intel oneTBB, OpenMP, and GNU Readline remain mandatory regardless of the optional integration settings.
