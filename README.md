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
   * *Role:* Acts as the first-line coarse filter. Because it is a compact 1D descriptor, it can be indexed instantly using FAISS to eliminate pairs whose radial distance distributions diverge beyond tolerance.

2. **SDV (Sorted Distance Vector):**
   * *Definition:* A pairwise distance descriptor of size $N(N-1)$, capturing all inter-atomic distances within the point cloud, sorted sequentially.
   * *Role:* Provides a stricter rigid-body invariant. Pairs that survive the SRV threshold are evaluated against SDV metrics.

3. **PDD (Pointwise Distance Distribution):**
   * *Definition:* For every vertex, we compute its distance to all other vertices and sort them, forming an $N \times N$ matrix. 
   * *Role:* Compared using **Bottleneck distance** (or Earth Mover's Distance) to capture local neighborhood consistency across the structure.

4. **The Aligner (Rigid Transformation & Bottleneck Optimization):**
   * *Definition:* The ultimate geometric verification stage. It constructs a local orthonormal coordinate frame from the most non-degenerate triangle in cloud $A$ and evaluates ranked candidate frames in cloud $B$.
   * *Role:* Solves the exact 3D rigid transformation (rotation + translation), computes the true Bottleneck distance via bitwise bipartite matching, and optionally performs sub-angstrom refinement using the Kabsch algorithm. *(See the deep-dive section below for full algorithmic details.)*

---

## 🔬 Deep Dive: How The Aligner Works

When candidate molecules survive the fast descriptor filters (`SRV`, `SDV`, and `PDD`), `CloudAlignmentEvaluator` executes a multi-phase geometric alignment to prove equivalence under rigid motion and noise:

### 1. Centroid Normalization
Both raw point clouds are translated so their geometric centroids ($c_A, c_B$) sit at the origin $(0,0,0)$. This decouples translational offsets from rotational alignment, allowing pure rotational frames to be compared directly.

### 2. Non-Degenerate Reference Triangle & Orthonormal Frame ($Fa$)
To lock in a stable 3D coordinate basis without collinearity issues, the algorithm scans all vertex pairs $(i, j)$ in Cloud $A$ to find the pair $(a_1, a_2)$ that maximizes the area squared of the triangle formed with the origin:
$$\text{Area}^2 = \|\mathbf{a}_1 \times \mathbf{a}_2\|^2$$
Using $\mathbf{a}_1$ and $\mathbf{a}_2$, a right-handed orthonormal basis ($Fa$) is generated via cross products.

### 3. Candidate Pair Ranking in Cloud $B$
The aligner exhaustively evaluates candidate pairs $(b_1, b_2)$ in Cloud $B$ (enforcing atomic element symbol matching when `symbol=1` is enabled). Candidates are scored and sorted by how closely their edge lengths, vertex radial norms, and triangle surface areas match $(a_1, a_2)$, allowing top-ranked candidates to be tested first.

### 4. Frame Rotation, Bitwise Bipartite Matching & Bottleneck Evaluation
For each candidate pair in $B$, a local basis $Fb$ is built, uniquely determining the candidate rotation matrix $M = Fb \cdot Fa^T$. 
* Cloud $A$ is rotated into Cloud $B$'s space to populate a squared-distance cost matrix.
* Fast bitwise adjacency masks (`__builtin_ctzll`) and augmenting path searches check if a valid 1-to-1 atomic assignment exists within the current distance threshold.
* Valid assignments are scored using the exact **Bottleneck distance**, updating the global upper bound and enabling early-exit pruning when `exact=0` or `accept` thresholds are met.

### 5. Sub-Angstrom Kabsch Refinement
When `refine > 0` is configured, the winning 1-to-1 atomic matching is passed into the **Kabsch algorithm**. Using quaternion eigenvalue decomposition (via Jacobi rotations on a $4 \times 4$ covariance matrix), it computes the optimal rigid rotation across *all* atoms simultaneously, iteratively lowering the Bottleneck distance even in the presence of Gaussian coordinate noise.

---

## 🖥️ Interactive Visualization & CLI (Polyscope + ImGui)

`ScientificDB` features a rich interactive command-line interface paired with a live **Polyscope 3D visualization window** and custom **ImGui** panels:

* **Real-time Inspection:** Load single files or folders (`loadXyz`), render molecules as colored 3D point clouds (`view <index>`), and manipulate them with interactive transform gizmos.
* **Alignment Overlays:** Inspect exact 1-to-1 atomic correspondence lines (`show_match`) and snap aligned structures onto one another (`apply_align`).
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
