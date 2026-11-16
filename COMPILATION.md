# Building ScientificDB

`ScientificDB` is a C++17 application that uses **CMake** to manage dependencies, automatically fetching **Chemfiles** and **Polyscope** while linking against system-installed **CUDA**, **FAISS**, **OpenBLAS**, **Intel TBB**, and **OpenMP**.

By default, **MariaDB/MySQL support is disabled** (`ENABLE_MYSQL=OFF`), allowing you to compile and run the project cleanly in standalone/offline mode without needing a database server or client libraries installed.

---

## 1. System Prerequisites

Before compiling, ensure you have a C++17-compatible compiler (`g++` or `clang++`), `git`, and **CMake 3.15+** installed, along with the following system libraries:

* **CUDA Toolkit** (headers in `/usr/local/cuda/include`, libraries in `/usr/local/cuda/lib64`)
* **FAISS** (headers in `/usr/local/include`, libraries in `/usr/local/lib64`)
* **OpenBLAS**
* **Intel oneTBB** (`tbb`)
* **OpenMP**
* **GNU Readline** (`readline`)

### Example Package Installation (Fedora / RHEL)
```bash
sudo dnf install cmake gcc-c++ git tbb-devel openblas-devel readline-devel
