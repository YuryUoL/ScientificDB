# ScientificDB 🚀🧬

**ScientificDB** is a high-performance C++17 computational engine and database-backed repository designed for indexing, visualizing, and performing rapid similarity searches on 3D molecular and crystal point clouds. 

Built to handle massive datasets with millions of structures, it integrates **FAISS** vector indexing, **Polyscope** real-time 3D visualization, multi-threaded CPU processing (via **OpenMP** and **Intel TBB**), and optional **MySQL/MariaDB** persistent storage.

---

## 🎯 Project Vision & Roadmap

In chemical and materials science research, duplicate or structurally redundant molecules frequently pollute datasets, skewing quantum chemical calculations, machine learning training sets, and published papers. 

* **Milestone 1 (Current Focus):** Storage, indexing, and rapid duplicate detection for molecular data (labeled 3D point clouds).
* **Milestone 2 (Future Goal):** Expansion to crystal data (labeled periodic 3D point clouds under lattice symmetry).

---

## 🗄️ Why Database Integration (MySQL / MariaDB)?

Chemists often manage disparate `.xyz` files scattered across local directories. `ScientificDB` integrates natively with **MySQL and MariaDB** to solve this:
* **Centralized Repositories:** Store hundreds of thousands of molecules in a single robust database using compressed binary blobs alongside precomputed descriptor vectors.
* **ACID Transactions & Bulk Pushing:** Push large datasets into the database efficiently using transaction batches (`db_push`, `db_pull`), ensuring data integrity and rapid retrieval.
* **Collaborative Research:** Clean your lab's computational pipeline by querying out invalid, corrupted, or redundant molecules before publishing research papers.

---

## 📐 Mathematical Invariants & Descriptors

A core challenge in structural chemistry is identifying whether two point clouds are identical up to **small noise ($\epsilon$) and rigid motion** (translation and rotation), without getting bogged down by expensive brute-force coordinate alignments. 

`ScientificDB` uses a hierarchical multi-stage filtering pipeline powered by geometric invariants:

1. **SRV (Sorted Radial Vector):**
   * *Definition:* For every atom, we compute its Euclidean distance to the molecular centroid. This vector of size $N$ is then sorted in ascending order.
   * *Role:* Acts as the first-line coarse filter. Because it is a compact $1D$ descriptor, it can be indexed instantly using FAISS to eliminate pairs whose radial distance distributions diverge beyond tolerance.

2. **SDV (Sorted Distance Vector):**
   * *Definition:* A pairwise distance descriptor of size $N(N-1)$, capturing all inter-atomic distances within the point cloud, sorted sequentially.
   * *Role:* Provides a stricter rigid-body invariant. Pairs that survive the SRV threshold are evaluated against SDV metrics.

3. **PDD (Pointwise Distance Distribution):**
   * *Definition:* For every vertex, we compute its distance to all other vertices and sort them, forming an $N \times N$ matrix. 
   * *Role:* Compared using **Bottleneck distance** (or Earth Mover's Distance) to capture local neighborhood consistency across the structure.

4. **The Aligner (Rigid Transformation & Bottleneck Optimization):**
   * *Definition:* The ultimate verification stage. It identifies the two most non-degenerate points $(a, b)$ in cloud $A$ (forming a robust reference triangle with the centroid $[0, a, b]$), and ranks corresponding candidate pairs $(c, d)$ in cloud $B$ based on edge lengths and triangle area matching.
   * *Role:* Because we operate in 3D, aligning these two vectors uniquely determines the rigid rotation. The algorithm computes the resulting Bottleneck distance as an upper bound, performing rapid early-exit or optimal minimization to find the exact matching and transformation matrix with extremely high probability under low noise.

---

## 🖥️ Interactive Visualization & CLI (Polyscope + ImGui)

`ScientificDB` features a rich interactive command-line interface paired with a live **Polyscope 3D visualization window** and custom **ImGui** panels:

* **Real-time Inspection:** Load single files or folders (`loadXyz`), render molecules as colored 3D point clouds (`view <index>`), and manipulate them with interactive transform gizmos.
* **Duplicate Network Graphs:** Visualize structural cluster graphs (`explore_clusters`, `show_graph`) to see network connectivity among duplicate molecular families.
* **Interactive Table Browser:** Inspect all loaded structures, atom counts, and metadata via the built-in ImGui Dataset Browser window.

---

## ⚙️ Quick Start & Compilation

Refer to [COMPILATION.md](COMPILATION.md) for detailed build instructions. 

**Standard Offline Build (No MySQL required):**
```bash
git clone https://github.com/your-username/ScientificDB.git
cd ScientificDB
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j$(nproc)
```

**Run the CLI Client:**
```bash
./ScientificDB
```
Type `help` inside the interactive prompt to explore available commands (`loadAndCompute`, `dup_check`, `pair_dist`, `db_connect`, etc.).

---

## 📜 License
Distributed under the **GPLv3** License. See `LICENSE` for more information.
