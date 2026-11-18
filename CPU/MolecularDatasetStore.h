#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include "Visualization/PolyscopeBridge.h"
#include "FaissIndexManager.h"
#include "Aligners/CloudAlignmentEvaluator.h"
#include "Aligners/AtomicCloudAlignmentEvaluator.h"
#include "Visualization/Matchingvisualizer.h"
#include "Visualization/AtomicMatchingViz.h"
#include "Duplicatechecker.h"
#include "Visualization/MolecularGraphViz.h"
#include <oneapi/tbb/parallel_sort.h>

struct MoleculeEntry
{
    size_t id;
    std::string filename;
    std::string commentLine;
    std::vector<AtomRecord> atoms;
    size_t numAtoms = 0;
};

struct AtomicCloudEntry
{
    size_t id;
    size_t parentMoleculeIdx;
    size_t atomIndex;
    std::string name;   // e.g., "water.xyz_H0"
    std::string symbol; // e.g., "H"
    std::vector<AtomRecord> atoms;
    size_t numAtoms = 0;

    // --- New Atomic Descriptors ---
    std::vector<float> sortedDistances;    // N-1 distances sorted ascending (excluding center atom at 0)
    std::vector<size_t> sortedAtomIndices; // Original atom indices sorted by distance to centroid
    bool descriptorsComputed = false;
};

class MolecularDatasetStore
{
public:
    MolecularDatasetStore() = default;
    ~MolecularDatasetStore() { clearAll(); }

bool computeAtomicAlignerDistance(size_t idxA, size_t idxB, size_t N, const AlignmentOptions& opt = AlignmentOptions())
    {
        if (idxA >= m_atomicEntries.size() || idxB >= m_atomicEntries.size())
        {
            std::cout << "[AtomicStore] Error: Index out of range.\n";
            return false;
        }

        const auto &acA = m_atomicEntries[idxA];
        const auto &acB = m_atomicEntries[idxB];

        if (!acA.descriptorsComputed || !acB.descriptorsComputed)
        {
            std::cout << "[AtomicStore] Error: Descriptors not computed. Run 'ComputeAtomicDescriptors' first.\n";
            return false;
        }

        // Cloud A subset: Take N closest atoms (EXCLUDING the centroid at index 0)
        // sortedAtomIndices already excludes the center atom and is sorted by distance ascending.
        std::vector<size_t> idxA_subset;
        size_t limitA = (N > 0) ? std::min(N, acA.sortedAtomIndices.size()) : acA.sortedAtomIndices.size();
        for (size_t i = 0; i < limitA; ++i)
        {
            idxA_subset.push_back(acA.sortedAtomIndices[i]);
        }

        if (idxA_subset.empty())
        {
            std::cout << "[AtomicStore] Error: Cloud A subset is empty (N=0 or no neighbor atoms).\n";
            return false;
        }

        // Cloud B subset: Take all atoms EXCEPT the centroid (skip index 0)
        std::vector<size_t> idxB_subset;
        for (size_t j = 0; j < acB.atoms.size(); ++j)
        {
            if (j == 0) continue; // Exclude centroid
            idxB_subset.push_back(j);
        }

        if (idxB_subset.size() < idxA_subset.size())
        {
            std::cout << "[AtomicStore] Error: Cloud B subset size (" << idxB_subset.size() 
                      << ") is smaller than Cloud A subset size (" << idxA_subset.size() << ").\n";
            return false;
        }

        // Convert atoms to Point3F
        std::vector<CloudAlignmentEvaluator::Point3F> cloudA(acA.atoms.size());
        std::vector<std::string> symbolsA(acA.atoms.size());
        for (size_t i = 0; i < acA.atoms.size(); ++i)
        {
            cloudA[i] = {static_cast<float>(acA.atoms[i].x), static_cast<float>(acA.atoms[i].y), static_cast<float>(acA.atoms[i].z)};
            symbolsA[i] = acA.atoms[i].symbol;
        }

        std::vector<CloudAlignmentEvaluator::Point3F> cloudB(acB.atoms.size());
        std::vector<std::string> symbolsB(acB.atoms.size());
        for (size_t i = 0; i < acB.atoms.size(); ++i)
        {
            cloudB[i] = {static_cast<float>(acB.atoms[i].x), static_cast<float>(acB.atoms[i].y), static_cast<float>(acB.atoms[i].z)};
            symbolsB[i] = acB.atoms[i].symbol;
        }

        AlignmentOptions localOpt = opt;
        if (localOpt.requireSameSymbol)
        {
            localOpt.symbolsA = &symbolsA;
            localOpt.symbolsB = &symbolsB;
        }

        auto tStart = std::chrono::high_resolution_clock::now();
        SubsetAlignmentResult res = AtomicCloudAlignmentEvaluator::alignSubsetAndEvaluate(
            cloudA, idxA_subset, cloudB, idxB_subset, localOpt);
        auto tEnd = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        if (!res.success)
        {
            std::cout << "[AtomicAligner] Alignment failed or constraints not met.\n";
            return false;
        }

        AtomicMatchingViz::store(idxA, idxB, acA.name, acB.name, res, idxA_subset, idxB_subset);

        std::cout << "\n=======================================================\n";
        std::cout << "  ATOMIC CLOUD ALIGNMENT & MATCHING (aligner - NO CENTROIDS)\n";
        std::cout << "=======================================================\n";
        std::cout << "  Atomic Cloud A   : " << idxA << " (" << acA.name << ")\n";
        std::cout << "  Atomic Cloud B   : " << idxB << " (" << acB.name << ")\n";
        std::cout << "  Subset Size A    : " << idxA_subset.size() << " neighbor atoms (Centroid excluded)\n";
        std::cout << "  Subset Size B    : " << idxB_subset.size() << " neighbor atoms (Centroid excluded)\n";
        std::cout << "  Bottleneck Dist  : " << std::fixed << std::setprecision(6) << res.bestBottleneck << "\n";
        std::cout << "  Time Elapsed     : " << std::fixed << std::setprecision(3) << elapsedMs << " ms\n";
        std::cout << "-------------------------------------------------------\n";
        std::cout << "  Atom Mapping     : [";
        for (size_t i = 0; i < res.matching.size(); ++i)
        {
            std::cout << res.matching[i] << (i + 1 < res.matching.size() ? ", " : "");
        }
        std::cout << "]\n";
        std::cout << "=======================================================\n";
        std::cout << "  [AtomicMatching] Stored. Use 'apply_atomic_align " << idxA << " " << idxB << "' to rotate A into B.\n\n";
        return true;
    }

    // --- Compute Atomic Descriptors (SRV-analog & Sorted Index Array) ---
    void computeAtomicDescriptors()
    {
        if (m_atomicEntries.empty())
        {
            std::cout << "[AtomicStore] Error: No atomic clouds generated. Run 'expand_atoms' first.\n";
            return;
        }

        size_t totalClouds = m_atomicEntries.size();
        std::cout << "[AtomicStore] Computing atomic descriptors for " << totalClouds << " atomic clouds in parallel...\n";
        auto tStart = std::chrono::high_resolution_clock::now();

        std::atomic<size_t> completedClouds{0};

#pragma omp parallel for schedule(dynamic, 64)
        for (long long idx = 0; idx < static_cast<long long>(totalClouds); ++idx)
        {
            auto &ac = m_atomicEntries[idx];

            struct AtomDistPair
            {
                size_t originalIdx;
                float dist;
            };
            std::vector<AtomDistPair> pairs;
            pairs.reserve(ac.atoms.size() > 0 ? ac.atoms.size() - 1 : 0);

            for (size_t j = 0; j < ac.atoms.size(); ++j)
            {
                float x = static_cast<float>(ac.atoms[j].x);
                float y = static_cast<float>(ac.atoms[j].y);
                float z = static_cast<float>(ac.atoms[j].z);
                float dist = std::sqrt(x * x + y * y + z * z);

                // Avoid 0 distance (skip the center atom sitting at the origin)
                if (dist < 1e-6f)
                {
                    continue;
                }
                pairs.push_back({j, dist});
            }

            // Sort by distance ascending
            std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b)
                      { return a.dist < b.dist; });

            ac.sortedDistances.clear();
            ac.sortedAtomIndices.clear();
            ac.sortedDistances.reserve(pairs.size());
            ac.sortedAtomIndices.reserve(pairs.size());

            for (const auto &p : pairs)
            {
                ac.sortedDistances.push_back(p.dist);
                ac.sortedAtomIndices.push_back(p.originalIdx);
            }
            ac.descriptorsComputed = true;

            size_t done = ++completedClouds;
            if (done % 500 == 0 || done == totalClouds)
            {
#pragma omp critical
                {
                    printProgressBar(done, totalClouds);
                }
            }
        }

        auto tEnd = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        std::cout << "\n[AtomicStore] Successfully computed atomic descriptors in "
                  << std::fixed << std::setprecision(2) << elapsedMs << " ms.\n";
    }

    // --- Compute L_inf distance between two atomic clouds with parameter N ---
    float computeAtomicPairDistance(size_t idxA, size_t idxB, const std::string &metric, size_t N = 0) const
    {
        if (idxA >= m_atomicEntries.size() || idxB >= m_atomicEntries.size())
        {
            std::cout << "[AtomicStore] Error: Atomic cloud index out of range.\n";
            return -1.0f;
        }

        const auto &acA = m_atomicEntries[idxA];
        const auto &acB = m_atomicEntries[idxB];

        if (!acA.descriptorsComputed || !acB.descriptorsComputed)
        {
            std::cout << "[AtomicStore] Error: Atomic descriptors not computed. Run 'ComputeAtomicDescriptors' first.\n";
            return -1.0f;
        }

        if (metric == "aligner")
        {
            std::cout << "[AtomicStore] Aligner metric for atomic clouds is not yet supported.\n";
            return -1.0f;
        }

        if (metric == "srv" || metric == "default")
        {
            size_t lenA = acA.sortedDistances.size();
            size_t lenB = acB.sortedDistances.size();
            size_t maxAvailable = std::min(lenA, lenB);

            // If N is specified, cap it by the max available extent of them
            size_t limit = (N > 0) ? std::min(N, maxAvailable) : maxAvailable;

            if (limit == 0)
            {
                std::cout << "[AtomicStore] Warning: Comparison length is 0.\n";
                return 0.0f;
            }

            float maxDiff = 0.0f;
            for (size_t i = 0; i < limit; ++i)
            {
                float diff = std::fabs(acA.sortedDistances[i] - acB.sortedDistances[i]);
                if (diff > maxDiff)
                {
                    maxDiff = diff;
                }
            }
            return maxDiff;
        }

        std::cout << "[AtomicStore] Unknown metric: " << metric << "\n";
        return -1.0f;
    }

    // --- Atomic Neighborhood Cloud Expansion & Management ---
   void generateAllAtomicClouds()
    {
        m_atomicEntries.clear();
        size_t totalMols = m_entries.size();
        if (totalMols == 0)
        {
            std::cout << "[AtomicStore] No molecules loaded to expand.\n";
            return;
        }

        std::cout << "[AtomicStore] Expanding " << totalMols << " molecules into atomic neighborhoods in parallel...\n";
        auto tStart = std::chrono::high_resolution_clock::now();

        // Temporary thread-local storage containers to prevent race conditions during parallel push_back
        int maxThreads = omp_get_max_threads();
        std::vector<std::vector<AtomicCloudEntry>> threadPoolEntries(maxThreads);
        std::atomic<size_t> completedMols{0};

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            size_t localNextId = tid * 10000000; // Offset thread IDs to avoid collision

#pragma omp for schedule(dynamic, 64)
            for (size_t molIdx = 0; molIdx < totalMols; ++molIdx)
            {
                const auto &mol = m_entries[molIdx];
                for (size_t i = 0; i < mol.atoms.size(); ++i)
                {
                    const auto &centerAtom = mol.atoms[i];

                    AtomicCloudEntry atomicCloud;
                    atomicCloud.id = localNextId++;
                    atomicCloud.parentMoleculeIdx = molIdx;
                    atomicCloud.atomIndex = i;
                    atomicCloud.symbol = centerAtom.symbol;
                    atomicCloud.name = mol.filename + "_" + centerAtom.symbol + std::to_string(i);

                    atomicCloud.atoms.reserve(mol.atoms.size());
                    for (size_t j = 0; j < mol.atoms.size(); ++j)
                    {
                        AtomRecord shifted = mol.atoms[j];
                        shifted.x -= centerAtom.x;
                        shifted.y -= centerAtom.y;
                        shifted.z -= centerAtom.z;
                        atomicCloud.atoms.push_back(shifted);
                    }
                    atomicCloud.numAtoms = atomicCloud.atoms.size();
                    threadPoolEntries[tid].push_back(std::move(atomicCloud));
                }

                size_t done = ++completedMols;
                if (done % 50 == 0 || done == totalMols)
                {
#pragma omp critical
                    {
                        printProgressBar(done, totalMols);
                    }
                }
            }
        }

   // Merge thread-local results back into m_atomicEntries sequentially
        for (int tid = 0; tid < maxThreads; ++tid)
        {
            for (auto &ac : threadPoolEntries[tid])
            {
                m_atomicEntries.push_back(std::move(ac));
            }
        }

        // Parallel sort using Intel TBB (preserves file order and atom index order)
        tbb::parallel_sort(m_atomicEntries.begin(), m_atomicEntries.end(), [](const auto &a, const auto &b) {
            if (a.parentMoleculeIdx != b.parentMoleculeIdx)
                return a.parentMoleculeIdx < b.parentMoleculeIdx;
            return a.atomIndex < b.atomIndex;
        });

        // Reassign clean contiguous IDs after sorting
        for (size_t i = 0; i < m_atomicEntries.size(); ++i)
        {
            m_atomicEntries[i].id = i;
        }
        m_nextAtomicId = m_atomicEntries.size();

        auto tEnd = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        std::cout << "\n[AtomicStore] Successfully generated and sorted " << m_atomicEntries.size()
                  << " atomic clouds across " << totalMols << " molecules in "
                  << std::fixed << std::setprecision(2) << elapsedMs << " ms.\n";
    }

    void generateAtomicCloudsFor(size_t molIdx)
    {
        if (molIdx >= m_entries.size())
        {
            std::cout << "[AtomicStore] Error: Molecule index " << molIdx << " out of range.\n";
            return;
        }
        const auto &mol = m_entries[molIdx];

        for (size_t i = 0; i < mol.atoms.size(); ++i)
        {
            const auto &centerAtom = mol.atoms[i];

            AtomicCloudEntry atomicCloud;
            atomicCloud.id = m_nextAtomicId++;
            atomicCloud.parentMoleculeIdx = molIdx;
            atomicCloud.atomIndex = i;
            atomicCloud.symbol = centerAtom.symbol;
            atomicCloud.name = mol.filename + "_" + centerAtom.symbol + std::to_string(i);

            atomicCloud.atoms.reserve(mol.atoms.size());
            // Shift coordinates precisely so the chosen central atom is at (0,0,0)
            for (size_t j = 0; j < mol.atoms.size(); ++j)
            {
                AtomRecord shifted = mol.atoms[j];
                shifted.x -= centerAtom.x;
                shifted.y -= centerAtom.y;
                shifted.z -= centerAtom.z;
                atomicCloud.atoms.push_back(shifted);
            }
            atomicCloud.numAtoms = atomicCloud.atoms.size();
            m_atomicEntries.push_back(std::move(atomicCloud));
        }
    }

    void clearAtomicClouds()
    {
        // Quietly remove ONLY the atomic clouds that were actually rendered in Polyscope (no spam)
        for (const auto &ac : m_atomicEntries)
        {
            std::string cloudName = "Molecule_" + std::to_string(ac.id + 1000000) + "_" + ac.name;
            if (polyscope::hasPointCloud(cloudName))
            {
                polyscope::removePointCloud(cloudName, true);
            }
        }
        m_atomicEntries.clear();
        m_nextAtomicId = 0;
        std::cout << "[AtomicStore] Cleared all atomic neighborhood clouds.\n";
    }

    size_t atomicSize() const { return m_atomicEntries.size(); }

    std::string getAtomicName(size_t index) const
    {
        if (index >= m_atomicEntries.size())
            return "";
        return m_atomicEntries[index].name;
    }

    size_t getAtomicAtomCount(size_t index) const
    {
        if (index >= m_atomicEntries.size())
            return 0;
        return m_atomicEntries[index].numAtoms;
    }

    bool visualizeAtomicIndex(size_t index) const
    {
        if (index >= m_atomicEntries.size())
        {
            std::cout << "[AtomicStore] Error: Atomic index " << index << " out of range.\n";
            return false;
        }
        const auto &ac = m_atomicEntries[index];

        // Pass false for autoCenter so the central atom stays at (0,0,0)
        PolyscopeBridge::visualizeMolecule(ac.id + 1000000, ac.name, ac.atoms, false);

        std::cout << "[Polyscope] Rendered atomic neighborhood (centered on atom): " << ac.name << "\n";
        return true;
    }

    bool unviewAtomicIndex(size_t index) const
    {
        if (index >= m_atomicEntries.size())
            return false;
        const auto &ac = m_atomicEntries[index];
        PolyscopeBridge::removeVisualization(ac.id + 1000000, ac.name);
        return true;
    }

    void ensureDescriptorsUpToDate()
    {
        if (!m_descriptorsDirty && m_faissManager.descriptorCount() == m_entries.size())
            return;

        m_faissManager.syncWithEntries(
            m_entries.size(),
            [this](size_t i) -> const std::vector<AtomRecord> &
            { return m_entries[i].atoms; });
        m_descriptorsDirty = false;
    }

    bool addFile(const std::string &filepath)
    {
        std::ifstream file(filepath);
        if (!file.is_open())
        {
            return false;
        }

        std::string line;
        if (!std::getline(file, line))
            return false;

        int numAtoms = 0;
        try
        {
            numAtoms = std::stoi(line);
        }
        catch (...)
        {
            return false;
        }

        if (!std::getline(file, line))
            return false;
        std::string comment = line;

        MoleculeEntry mol;
        mol.id = m_nextId++;
        mol.filename = std::filesystem::path(filepath).filename().string();
        mol.commentLine = comment;
        mol.atoms.reserve(numAtoms);

        for (int i = 0; i < numAtoms; ++i)
        {
            if (!std::getline(file, line))
                break;
            std::istringstream iss(line);
            AtomRecord atom;
            if (iss >> atom.symbol >> atom.x >> atom.y >> atom.z)
            {
                if (!(iss >> atom.charge))
                {
                    atom.charge = 0.0;
                }
                mol.atoms.push_back(atom);
            }
        }

        mol.numAtoms = mol.atoms.size();
        m_entries.push_back(std::move(mol));
        m_descriptorsDirty = true;
        return true;
    }

    bool addStringContent(const std::string &filename, const std::string &content)
    {
        std::istringstream file(content);
        std::string line;
        if (!std::getline(file, line))
            return false;

        int numAtoms = 0;
        try
        {
            numAtoms = std::stoi(line);
        }
        catch (...)
        {
            return false;
        }

        if (!std::getline(file, line))
            return false;
        std::string comment = line;

        MoleculeEntry mol;
        mol.id = m_nextId++;
        mol.filename = filename;
        mol.commentLine = comment;
        mol.atoms.reserve(numAtoms);

        for (int i = 0; i < numAtoms; ++i)
        {
            if (!std::getline(file, line))
                break;
            std::istringstream iss(line);
            AtomRecord atom;
            if (iss >> atom.symbol >> atom.x >> atom.y >> atom.z)
            {
                if (!(iss >> atom.charge))
                {
                    atom.charge = 0.0;
                }
                mol.atoms.push_back(atom);
            }
        }

        mol.numAtoms = mol.atoms.size();
        m_entries.push_back(std::move(mol));
        m_descriptorsDirty = true;
        return true;
    }

    size_t addSyntheticEntry(const std::string &filename, const std::string &comment, const std::vector<AtomRecord> &atoms)
    {
        MoleculeEntry mol;
        mol.id = m_nextId++;
        mol.filename = filename;
        mol.commentLine = comment;
        mol.atoms = atoms;
        mol.numAtoms = atoms.size();
        m_entries.push_back(std::move(mol));
        m_descriptorsDirty = true;
        return m_entries.size();
    }

    std::vector<DupMatch> runDuplicateCheck(size_t queryIdx, const std::string &csvPath, const DupCheckParams &p, DupRunStats &st)
    {
        DuplicateChecker checker;
        if (queryIdx >= m_entries.size())
        {
            std::cout << "[XyzStore] Error: Query index out of range.\n";
            st = DupRunStats{};
            return {};
        }

        ensureDescriptorsUpToDate();

        const auto &qMol = m_entries[queryIdx];
        const auto *qDesc = m_faissManager.descriptor(queryIdx);
        if (!qDesc || !qDesc->computed)
        {
            std::cout << "[XyzStore] Error: Descriptors could not be computed.\n";
            st = DupRunStats{};
            return {};
        }

        std::vector<DupMatch> matches = checker.run(
            m_faissManager,
            queryIdx,
            *qDesc,
            qMol.atoms,
            [this](size_t i) -> const std::vector<AtomRecord> &
            { return m_entries[i].atoms; },
            [this](size_t i) -> std::string
            { return m_entries[i].commentLine; },
            p,
            st);

        for (const auto &m : matches)
        {
            const auto &molA = m_entries[m.queryIdx];
            const auto &molB = m_entries[m.matchIdx];
            MatchingViz::store(molA.id, molB.id, molA.filename, molB.filename, m.align);
        }

        if (!csvPath.empty() && !matches.empty())
        {
            std::ofstream csv(csvPath);
            if (csv.is_open())
            {
                csv << "MolA,MolB,Atoms,align,pdd,sdv,srv,mismatch\n";
                for (const auto &m : matches)
                {
                    csv << m.queryIdx << ","
                        << m.matchIdx << ","
                        << m.numAtoms << ","
                        << std::fixed << std::setprecision(6)
                        << m.alignBottleneck << ","
                        << m.pddBottleneck << ","
                        << m.sdvLinf << ","
                        << m.srvLinf << ","
                        << m.speciesMismatches << "\n";
                }
                std::cout << "[XyzStore] Successfully wrote " << matches.size() << " duplicate pairs to " << csvPath << "\n";
            }
        }

        return matches;
    }

    std::vector<DupMatch> runBatchDuplicateCheck(const std::string &csvPath, const DupCheckParams &p, DupRunStats &st)
    {
        DuplicateChecker checker;
        size_t totalMols = m_entries.size();
        if (totalMols < 2)
        {
            std::cout << "[XyzStore] Error: Not enough molecules loaded for batch duplicate check.\n";
            st = DupRunStats{};
            return {};
        }

        ensureDescriptorsUpToDate();

        std::vector<DupMatch> matches = checker.runAll(
            m_faissManager,
            totalMols,
            [this](size_t i) -> const std::vector<AtomRecord> &
            { return m_entries[i].atoms; },
            [this](size_t i) -> std::string
            { return m_entries[i].commentLine; },
            p,
            st);

        for (const auto &m : matches)
        {
            const auto &molA = m_entries[m.queryIdx];
            const auto &molB = m_entries[m.matchIdx];
            MatchingViz::store(molA.id, molB.id, molA.filename, molB.filename, m.align);
        }

        if (!csvPath.empty())
        {
            std::ofstream csv(csvPath);
            if (csv.is_open())
            {
                csv << "MolA,MolB,Atoms,align,pdd,sdv,srv,mismatch\n";
                for (const auto &m : matches)
                {
                    csv << m.queryIdx << ","
                        << m.matchIdx << ","
                        << m.numAtoms << ","
                        << std::fixed << std::setprecision(6)
                        << m.alignBottleneck << ","
                        << m.pddBottleneck << ","
                        << m.sdvLinf << ","
                        << m.srvLinf << ","
                        << m.speciesMismatches << "\n";
                }
                std::cout << "[XyzStore] Successfully wrote " << matches.size() << " duplicate pairs to " << csvPath << "\n";
            }
        }

        m_lastBatchMatches = matches;
        return matches;
    }

    size_t addFolder(const std::string &folderpath)
    {
        namespace fs = std::filesystem;
        if (!fs::exists(folderpath) || !fs::is_directory(folderpath))
        {
            std::cerr << "[XyzStore] Invalid directory path: " << folderpath << "\n";
            return 0;
        }

        std::vector<std::string> filePaths;
        for (const auto &entry : fs::directory_iterator(folderpath))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".xyz")
            {
                filePaths.push_back(entry.path().string());
            }
        }

        std::sort(filePaths.begin(), filePaths.end());

        size_t totalFiles = filePaths.size();
        if (totalFiles == 0)
        {
            std::cout << "[XyzStore] No .xyz files found in directory.\n";
            return 0;
        }

        std::cout << "[XyzStore] Loading " << totalFiles << " sorted files from folder...\n";

        size_t count = 0;
        auto tStart = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < totalFiles; ++i)
        {
            if (addFile(filePaths[i]))
            {
                count++;
            }

            if ((i + 1) % 50 == 0 || (i + 1) == totalFiles)
            {
                printProgressBar(i + 1, totalFiles);
            }
        }

        auto tEnd = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        std::cout << "\n[XyzStore] Successfully loaded " << count << " .xyz files into host RAM in "
                  << std::fixed << std::setprecision(2) << elapsedMs << " ms.\n";
        return count;
    }

    bool searchKNN(size_t index, const std::string &kStr, const std::string &metricType, const std::string &csvPath = "", bool restrictToLast = false)
    {
        ensureDescriptorsUpToDate();
        return m_faissManager.searchKNN(index, kStr, metricType, csvPath, restrictToLast);
    }

    void printDistanceDistribution() const
    {
        m_faissManager.printDistanceDistribution();
    }

    bool printPairAlignmentWithStats(size_t idxA, size_t idxB, AlignmentOptions opt = AlignmentOptions())
    {
        if (idxA >= m_entries.size() || idxB >= m_entries.size())
        {
            std::cout << "[XyzStore] Error: Index out of range.\n";
            return false;
        }

        const auto &molA = m_entries[idxA];
        const auto &molB = m_entries[idxB];

        if (molA.numAtoms != molB.numAtoms || molA.numAtoms == 0)
        {
            std::cout << "[XyzStore] Error: Molecules must have an equal and non-zero number of atoms for alignment.\n";
            return false;
        }

        std::vector<CloudAlignmentEvaluator::Point3F> cloudA(molA.numAtoms);
        std::vector<std::string> symbolsA(molA.numAtoms);
        for (size_t i = 0; i < molA.numAtoms; ++i)
        {
            cloudA[i] = {static_cast<float>(molA.atoms[i].x), static_cast<float>(molA.atoms[i].y), static_cast<float>(molA.atoms[i].z)};
            symbolsA[i] = molA.atoms[i].symbol;
        }

        std::vector<CloudAlignmentEvaluator::Point3F> cloudB(molB.numAtoms);
        std::vector<std::string> symbolsB(molB.numAtoms);
        for (size_t i = 0; i < molB.numAtoms; ++i)
        {
            cloudB[i] = {static_cast<float>(molB.atoms[i].x), static_cast<float>(molB.atoms[i].y), static_cast<float>(molB.atoms[i].z)};
            symbolsB[i] = molB.atoms[i].symbol;
        }

        if (opt.requireSameSymbol)
        {
            opt.symbolsA = &symbolsA;
            opt.symbolsB = &symbolsB;
        }

        auto tStart = std::chrono::high_resolution_clock::now();
        AlignmentResult res = CloudAlignmentEvaluator::alignAndEvaluate(cloudA, cloudB, opt);
        auto tEnd = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        if (!res.success)
        {
            std::cout << "[CloudAlignment] Alignment failed or exceeded vertex limit.\n";
            return false;
        }

        MatchingViz::store(molA.id, molB.id, molA.filename, molB.filename, res);

        std::cout << "\n=======================================================\n";
        std::cout << "  PAIRWISE RIGID ALIGNMENT & MATCHING (aligner)\n";
        std::cout << "=======================================================\n";
        std::cout << "  Molecule A Index : " << idxA << " (" << molA.filename << ")\n";
        std::cout << "  Molecule B Index : " << idxB << " (" << molB.filename << ")\n";
        std::cout << "  Bottleneck Dist  : " << std::fixed << std::setprecision(6) << res.bestBottleneck << "\n";
        std::cout << "  Time Elapsed     : " << std::fixed << std::setprecision(3) << elapsedMs << " ms\n";
        std::cout << "  Matching Pairs   : " << res.matching.size() << " atoms matched\n";
        std::cout << "-------------------------------------------------------\n";

        // Print matching in compressed array format: [0, 1, 2, 3, ...]
        std::cout << "  Atom Mapping     : [";
        for (size_t i = 0; i < res.matching.size(); ++i)
        {
            std::cout << res.matching[i] << (i + 1 < res.matching.size() ? ", " : "");
        }
        std::cout << "]\n";

        std::cout << "=======================================================\n";
        std::cout << "  [Matching] Stored. Use 'show_match " << idxA << " " << idxB << "' to display it.\n\n";
        return true;
    }

    bool visualizeIndex(size_t index) const
    {
        if (index >= m_entries.size())
        {
            std::cout << "[XyzStore] Error: Index " << index << " out of range (Total: " << m_entries.size() << ").\n";
            return false;
        }

        const auto &mol = m_entries[index];
        PolyscopeBridge::visualizeMolecule(mol.id, mol.filename, mol.atoms);
        std::cout << "[Polyscope] Successfully rendered molecule index " << index << " (" << mol.filename << ") as point cloud.\n";
        return true;
    }

    bool unviewIndex(size_t index) const
    {
        if (index >= m_entries.size())
        {
            std::cout << "[XyzStore] Error: Index " << index << " out of range (Total: " << m_entries.size() << ").\n";
            return false;
        }

        const auto &mol = m_entries[index];
        PolyscopeBridge::removeVisualization(mol.id, mol.filename);
        return true;
    }

    bool removeByIndex(size_t index)
    {
        if (index < m_entries.size())
        {
            MatchingViz::removeForMolecule(m_entries[index].id);
            size_t id = m_entries[index].id;
            m_entries.erase(m_entries.begin() + index);
            m_faissManager.removeEntryAt(index);
            m_descriptorsDirty = (m_faissManager.descriptorCount() != m_entries.size());
            std::cout << "[XyzStore] Removed molecule at index " << index << " (ID: " << id << ")\n";
            return true;
        }
        std::cout << "[XyzStore] Error: Index " << index << " out of range.\n";
        return false;
    }

    bool searchKNNForAll(const std::string &kStr, const std::string &metricType, const std::string &csvPath)
    {
        ensureDescriptorsUpToDate();
        return m_faissManager.searchKNNForAll(kStr, metricType, csvPath);
    }

    float computePairMetric(size_t idxA, size_t idxB, const std::string &metricType)
    {
        ensureDescriptorsUpToDate();
        return m_faissManager.computePairDistance(idxA, idxB, metricType);
    }

    void showAll(size_t limit = 25) const
    {
        std::cout << "\n=======================================================\n";
        std::cout << "          MOLECULAR DATASET STORE (HOST RAM)           \n";
        std::cout << "=======================================================\n";
        std::cout << "  Total Loaded Entries : " << m_entries.size() << "\n";
        std::cout << "-------------------------------------------------------\n";
        std::cout << std::left << std::setw(6) << "Idx"
                  << std::setw(8) << "ID"
                  << std::setw(28) << "Filename"
                  << std::setw(10) << "Atoms" << "\n";
        std::cout << "-------------------------------------------------------\n";

        size_t displayCount = std::min(m_entries.size(), limit);
        for (size_t i = 0; i < displayCount; ++i)
        {
            const auto &mol = m_entries[i];
            std::cout << std::left << std::setw(6) << i
                      << std::setw(8) << mol.id
                      << std::setw(28) << (mol.filename.length() > 25 ? mol.filename.substr(0, 22) + "..." : mol.filename)
                      << std::setw(10) << mol.numAtoms << "\n";
        }
        std::cout << "=======================================================\n\n";
    }

    void computeDescriptors()
    {
        m_faissManager.computeDescriptorsForAll(
            m_entries.size(),
            [this](size_t i) -> const std::vector<AtomRecord> &
            { return m_entries[i].atoms; });
        m_descriptorsDirty = false;
    }

    void flushDescriptors()
    {
        m_faissManager.clear();
        m_descriptorsDirty = true;
    }

    AlignmentResult computePairAlignmentWithStats(size_t idxA, size_t idxB, AlignmentOptions opt = AlignmentOptions())
    {
        if (idxA >= m_entries.size() || idxB >= m_entries.size())
        {
            return AlignmentResult{};
        }

        const auto &molA = m_entries[idxA];
        const auto &molB = m_entries[idxB];

        if (molA.numAtoms != molB.numAtoms || molA.numAtoms == 0)
        {
            return AlignmentResult{};
        }

        std::vector<CloudAlignmentEvaluator::Point3F> cloudA(molA.numAtoms);
        std::vector<std::string> symbolsA(molA.numAtoms);
        for (size_t i = 0; i < molA.numAtoms; ++i)
        {
            cloudA[i] = {static_cast<float>(molA.atoms[i].x), static_cast<float>(molA.atoms[i].y), static_cast<float>(molA.atoms[i].z)};
            symbolsA[i] = molA.atoms[i].symbol;
        }

        std::vector<CloudAlignmentEvaluator::Point3F> cloudB(molB.numAtoms);
        std::vector<std::string> symbolsB(molB.numAtoms);
        for (size_t i = 0; i < molB.numAtoms; ++i)
        {
            cloudB[i] = {static_cast<float>(molB.atoms[i].x), static_cast<float>(molB.atoms[i].y), static_cast<float>(molB.atoms[i].z)};
            symbolsB[i] = molB.atoms[i].symbol;
        }

        if (opt.requireSameSymbol)
        {
            opt.symbolsA = &symbolsA;
            opt.symbolsB = &symbolsB;
        }

        AlignmentResult res = CloudAlignmentEvaluator::alignAndEvaluate(cloudA, cloudB, opt);
        if (res.success)
        {
            MatchingViz::store(molA.id, molB.id, molA.filename, molB.filename, res);
        }
        return res;
    }

    void inspect(size_t index) const
    {
        if (index >= m_entries.size())
        {
            std::cout << "[XyzStore] Error: Index " << index << " out of range (Total: " << m_entries.size() << ").\n";
            return;
        }

        const auto &mol = m_entries[index];
        std::cout << "\n=======================================================\n";
        std::cout << "           MOLECULE INSPECTION [Index " << index << "]     \n";
        std::cout << "=======================================================\n";
        std::cout << "  Database ID    : " << mol.id << "\n";
        std::cout << "  Filename       : " << mol.filename << "\n";
        std::cout << "  Comment        : " << mol.commentLine << "\n";
        std::cout << "  Atom Count     : " << mol.numAtoms << "\n";
        std::cout << "-------------------------------------------------------\n";
        std::cout << std::left << std::setw(6) << "No."
                  << std::setw(10) << "Symbol"
                  << std::setw(14) << "X"
                  << std::setw(14) << "Y"
                  << std::setw(14) << "Z" << "\n";
        std::cout << "-------------------------------------------------------\n";

        size_t printLimit = std::min(mol.atoms.size(), size_t(100)); // Print up to 100 atoms
        for (size_t i = 0; i < printLimit; ++i)
        {
            const auto &atom = mol.atoms[i];
            std::cout << std::left << std::setw(6) << i
                      << std::setw(10) << atom.symbol
                      << std::setw(14) << std::fixed << std::setprecision(4) << atom.x
                      << std::setw(14) << atom.y
                      << std::setw(14) << atom.z << "\n";
        }
        if (mol.atoms.size() > printLimit)
        {
            std::cout << "  ... (showing first " << printLimit << " of " << mol.atoms.size() << " atoms)\n";
        }
        std::cout << "=======================================================\n";

        m_faissManager.printDescriptors(index);
    }

    bool setMatchVisible(size_t a, size_t b, bool v)
    {
        if (a >= m_entries.size() || b >= m_entries.size())
        {
            std::cout << "[Matching] Index out of range.\n";
            return false;
        }
        auto *m = MatchingViz::find(m_entries[a].id, m_entries[b].id);
        if (!m)
        {
            std::cout << "[Matching] No stored matching for this pair.\n";
            return false;
        }
        m->visible = v;
        return true;
    }

    void setAllMatchesVisible(bool v) { MatchingViz::setAllVisible(v); }

    bool removeMatch(size_t a, size_t b)
    {
        if (a >= m_entries.size() || b >= m_entries.size())
        {
            std::cout << "[Matching] Index out of range.\n";
            return false;
        }
        return MatchingViz::remove(m_entries[a].id, m_entries[b].id);
    }

    bool applyAlignment(size_t a, size_t b)
    {
        if (a >= m_entries.size() || b >= m_entries.size())
        {
            std::cout << "[Matching] Index out of range.\n";
            return false;
        }
        return MatchingViz::applyAlignment(m_entries[a].id, m_entries[b].id);
    }

    size_t size() const { return m_entries.size(); }

    void clearAll()
    {
        clearAtomicClouds();
        MatchingViz::clearAll();
        m_entries.clear();
        m_faissManager.clear();
        m_lastBatchMatches.clear();
        m_descriptorsDirty = false;
    }

    void exploreClusters()
    {
        if (m_lastBatchMatches.empty())
        {
            std::cout << "[XyzStore] Error: No recent batch duplicate check found. Run 'dup_check all' first.\n";
            return;
        }

        std::vector<std::string> filenames(m_entries.size());
        std::vector<size_t> atomCounts(m_entries.size());
        for (size_t i = 0; i < m_entries.size(); ++i)
        {
            filenames[i] = m_entries[i].filename;
            atomCounts[i] = m_entries[i].numAtoms;
        }

        MolecularGraphViz::buildClusters(m_lastBatchMatches, atomCounts);
    }

    void setGraphVisible(bool visible) const
    {
        MolecularGraphViz::setGraphVisible(visible);
    }

    void clearGraph() const
    {
        MolecularGraphViz::clearGraph();
    }

    std::string getFilename(size_t index) const
    {
        if (index >= m_entries.size())
            return "";
        return m_entries[index].filename;
    }

    std::string getComment(size_t index) const
    {
        if (index >= m_entries.size())
            return "";
        return m_entries[index].commentLine;
    }

    size_t getAtomCount(size_t index) const
    {
        if (index >= m_entries.size())
            return 0;
        return m_entries[index].numAtoms;
    }

    std::string getRawContent(size_t index) const
    {
        if (index >= m_entries.size())
            return "";
        const auto &mol = m_entries[index];
        std::ostringstream oss;
        oss << mol.numAtoms << "\n"
            << mol.commentLine << "\n";
        for (const auto &a : mol.atoms)
        {
            oss << a.symbol << " " << a.x << " " << a.y << " " << a.z;
            if (a.charge != 0.0)
                oss << " " << a.charge;
            oss << "\n";
        }
        return oss.str();
    }

    std::vector<float> getSDV(size_t index) const
    {
        const auto *desc = m_faissManager.descriptor(index);
        if (!desc || !desc->computed)
            return {};
        return desc->sdv;
    }

    std::vector<float> getSRV(size_t index) const
    {
        const auto *desc = m_faissManager.descriptor(index);
        if (!desc || !desc->computed)
            return {};
        return desc->srv;
    }

    std::vector<float> getPDD(size_t index) const
    {
        const auto *desc = m_faissManager.descriptor(index);
        if (!desc || !desc->computed)
            return {};
        return desc->pdd;
    }

private:
    std::vector<MoleculeEntry> m_entries;
    std::vector<AtomicCloudEntry> m_atomicEntries;
    size_t m_nextId = 0;
    size_t m_nextAtomicId = 0;
    FaissIndexManager m_faissManager;
    std::vector<DupMatch> m_lastBatchMatches;
    bool m_descriptorsDirty = false;

    void printProgressBar(size_t current, size_t total) const
    {
        int barWidth = 40;
        float progress = static_cast<float>(current) / total;
        int tensorWidth = static_cast<int>(barWidth * progress);

        std::cout << "\r[";
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < tensorWidth)
                std::cout << "=";
            else if (i == tensorWidth)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0f) << "% (" << current << "/" << total << ")" << std::flush;
    }
};