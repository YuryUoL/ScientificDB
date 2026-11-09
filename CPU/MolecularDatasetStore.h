#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include "PolyscopeBridge.h"
#include "FaissIndexManager.h"
#include "CloudAligmentEvaluator.h"
#include "Matchingvisualizer.h"

struct MoleculeEntry {
    size_t id;
    std::string filename;
    std::string commentLine;
    std::vector<AtomRecord> atoms;
    size_t numAtoms = 0;
};

class MolecularDatasetStore {
public:
    MolecularDatasetStore() = default;
    ~MolecularDatasetStore() { clearAll(); }

    bool addFile(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            return false;
        }

        std::string line;
        if (!std::getline(file, line)) return false;
        
        int numAtoms = 0;
        try {
            numAtoms = std::stoi(line);
        } catch (...) {
            return false;
        }

        if (!std::getline(file, line)) return false;
        std::string comment = line;

        MoleculeEntry mol;
        mol.id = m_nextId++;
        mol.filename = std::filesystem::path(filepath).filename().string();
        mol.commentLine = comment;
        mol.atoms.reserve(numAtoms);

        for (int i = 0; i < numAtoms; ++i) {
            if (!std::getline(file, line)) break;
            std::istringstream iss(line);
            AtomRecord atom;
            if (iss >> atom.symbol >> atom.x >> atom.y >> atom.z) {
                if (!(iss >> atom.charge)) {
                    atom.charge = 0.0;
                }
                mol.atoms.push_back(atom);
            }
        }

        mol.numAtoms = mol.atoms.size();
        m_entries.push_back(std::move(mol));
        return true;
    }

    size_t addFolder(const std::string& folderpath) {
        namespace fs = std::filesystem;
        if (!fs::exists(folderpath) || !fs::is_directory(folderpath)) {
            std::cerr << "[XyzStore] Invalid directory path: " << folderpath << "\n";
            return 0;
        }

        // 1. Collect and sort file paths alphabetically/numerically
        std::vector<std::string> filePaths;
        for (const auto& entry : fs::directory_iterator(folderpath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".xyz") {
                filePaths.push_back(entry.path().string());
            }
        }

        std::sort(filePaths.begin(), filePaths.end());

        size_t totalFiles = filePaths.size();
        if (totalFiles == 0) {
            std::cout << "[XyzStore] No .xyz files found in directory.\n";
            return 0;
        }

        std::cout << "[XyzStore] Loading " << totalFiles << " sorted files from folder...\n";

        size_t count = 0;
        auto tStart = std::chrono::high_resolution_clock::now();

        // 2. Load files with a progress bar update
        for (size_t i = 0; i < totalFiles; ++i) {
            if (addFile(filePaths[i])) {
                count++;
            }

            // Print progress bar every 50 files or at completion
            if ((i + 1) % 50 == 0 || (i + 1) == totalFiles) {
                printProgressBar(i + 1, totalFiles);
            }
        }

        auto tEnd = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        std::cout << "\n[XyzStore] Successfully loaded " << count << " .xyz files into host RAM in " 
                  << std::fixed << std::setprecision(2) << elapsedMs << " ms.\n";
        return count;
    }

    // [UPDATED] Pass string kStr (to support numbers like "10" or percentages like "10%") and restrictToLast flag for chaining
    bool searchKNN(size_t index, const std::string& kStr, const std::string& metricType, const std::string& csvPath = "", bool restrictToLast = false) const {
        return m_faissManager.searchKNN(index, kStr, metricType, csvPath, restrictToLast);
    }

    // [NEW] Added wrapper for printing distance distribution stats, histogram, and outliers
    void printDistanceDistribution() const {
        m_faissManager.printDistanceDistribution();
    }

    void printPairAlignment(size_t idxA, size_t idxB) const {
        if (idxA >= m_entries.size() || idxB >= m_entries.size()) {
            std::cout << "[XyzStore] Error: Index out of range.\n";
            return;
        }

        const auto& molA = m_entries[idxA];
        const auto& molB = m_entries[idxB];

        if (molA.numAtoms != molB.numAtoms || molA.numAtoms == 0) {
            std::cout << "[XyzStore] Error: Molecules must have an equal and non-zero number of atoms for alignment.\n";
            return;
        }

        std::vector<CloudAlignmentEvaluator::Point3F> cloudA(molA.numAtoms);
        for (size_t i = 0; i < molA.numAtoms; ++i) {
            cloudA[i] = {static_cast<float>(molA.atoms[i].x), static_cast<float>(molA.atoms[i].y), static_cast<float>(molA.atoms[i].z)};
        }

        std::vector<CloudAlignmentEvaluator::Point3F> cloudB(molB.numAtoms);
        for (size_t i = 0; i < molB.numAtoms; ++i) {
            cloudB[i] = {static_cast<float>(molB.atoms[i].x), static_cast<float>(molB.atoms[i].y), static_cast<float>(molB.atoms[i].z)};
        }

        std::vector<float> pddA, pddB;
        BottleneckEvaluator::computePDD(cloudA, pddA);
        BottleneckEvaluator::computePDD(cloudB, pddB);

        auto tStart = std::chrono::high_resolution_clock::now();
        AlignmentResult res = CloudAlignmentEvaluator::alignAndEvaluate(cloudA, cloudB, pddA, pddB);
        auto tEnd = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        if (!res.success) {
            std::cout << "[CloudAlignment] Alignment failed or exceeded vertex limit.\n";
            return;
        }

        MatchingViz::store(molA.id, molB.id, molA.filename, molB.filename, res);
        std::cout << "[Matching] Stored. Use 'view' on both molecules to see the lines.\n";

        std::cout << "\n=======================================================\n";
        std::cout << "          CLOUD ALIGNMENT & BOTTLENECK MATCHING        \n";
        std::cout << "=======================================================\n";
        std::cout << "  Molecule A Index : " << idxA << " (" << molA.filename << ")\n";
        std::cout << "  Molecule B Index : " << idxB << " (" << molB.filename << ")\n";
        std::cout << "  Best Trio in B   : b1 = " << res.bestB1 << ", b2 = " << res.bestB2 << "\n";
        std::cout << "  Bottleneck Dist  : " << std::fixed << std::setprecision(6) << res.bestBottleneck << "\n";
        std::cout << "  Time Elapsed     : " << std::fixed << std::setprecision(3) << elapsedMs << " ms\n";
        std::cout << "-------------------------------------------------------\n";
        std::cout << "  Rotation Matrix (3x3):\n";
        for (int r = 0; r < 3; ++r) {
            std::cout << "    [ " << std::setw(10) << res.rotation[r*3 + 0] 
                      << ", " << std::setw(10) << res.rotation[r*3 + 1] 
                      << ", " << std::setw(10) << res.rotation[r*3 + 2] << " ]\n";
        }
        std::cout << "  Translation Vector [tx, ty, tz]: [ " 
                  << res.translation[0] << ", " << res.translation[1] << ", " << res.translation[2] << " ]\n";
        std::cout << "-------------------------------------------------------\n";
        std::cout << "  Point-to-Point Matching (Cloud A -> Cloud B):\n";
        std::cout << std::left << std::setw(12) << "Index A" << std::setw(12) << "Index B" << "\n";
        std::cout << "-------------------------------------------------------\n";
        for (size_t i = 0; i < res.matching.size(); ++i) {
            std::cout << std::left << std::setw(12) << i << std::setw(12) << res.matching[i] << "\n";
        }
        std::cout << "=======================================================\n\n";
    }

    bool visualizeIndex(size_t index) const {
        if (index >= m_entries.size()) {
            std::cout << "[XyzStore] Error: Index " << index << " out of range (Total: " << m_entries.size() << ").\n";
            return false;
        }

        const auto& mol = m_entries[index];
        PolyscopeBridge::visualizeMolecule(mol.id, mol.filename, mol.atoms);
        std::cout << "[Polyscope] Successfully rendered molecule index " << index << " (" << mol.filename << ") as point cloud.\n";
        return true;
    }

    // Removes visual rendering of a molecule by index
    bool unviewIndex(size_t index) const {
        if (index >= m_entries.size()) {
            std::cout << "[XyzStore] Error: Index " << index << " out of range (Total: " << m_entries.size() << ").\n";
            return false;
        }

        const auto& mol = m_entries[index];
        PolyscopeBridge::removeVisualization(mol.id, mol.filename);
        return true;
    }

    bool removeByIndex(size_t index) {
        if (index < m_entries.size()) {
            MatchingViz::removeForMolecule(m_entries[index].id);
            size_t id = m_entries[index].id;
            m_entries.erase(m_entries.begin() + index);
            std::cout << "[XyzStore] Removed molecule at index " << index << " (ID: " << id << ")\n";
            return true;
        }
        std::cout << "[XyzStore] Error: Index " << index << " out of range.\n";
        return false;
    }

    bool searchKNNForAll(const std::string& kStr, const std::string& metricType, const std::string& csvPath) const {
        return m_faissManager.searchKNNForAll(kStr, metricType, csvPath);
    }
    float computePairMetric(size_t idxA, size_t idxB, const std::string& metricType) const {
    return m_faissManager.computePairDistance(idxA, idxB, metricType);
}

    void showAll(size_t limit = 25) const {
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
        for (size_t i = 0; i < displayCount; ++i) {
            const auto& mol = m_entries[i];
            std::cout << std::left << std::setw(6) << i
                      << std::setw(8) << mol.id
                      << std::setw(28) << (mol.filename.length() > 25 ? mol.filename.substr(0, 22) + "..." : mol.filename)
                      << std::setw(10) << mol.numAtoms << "\n";
        }
        std::cout << "=======================================================\n\n";
    }

    // Trigger parallel computation of SRD and SRV across all loaded molecules
    void computeDescriptors() {
        m_faissManager.computeDescriptorsForAll(
            m_entries.size(),
            [this](size_t i) -> const std::vector<AtomRecord>& { return m_entries[i].atoms; }
        );
    }

    // Flush/delete all descriptors and FAISS indices
    void flushDescriptors() {
        m_faissManager.clear();
    }

    void inspect(size_t index) const {
        if (index >= m_entries.size()) {
            std::cout << "[XyzStore] Error: Index " << index << " out of range (Total: " << m_entries.size() << ").\n";
            return;
        }

        const auto& mol = m_entries[index];
        std::cout << "\n=======================================================\n";
        std::cout << "           MOLECULE INSPECTION [Index " << index << "]     \n";
        std::cout << "=======================================================\n";
        std::cout << "  Database ID    : " << mol.id << "\n";
        std::cout << "  Filename       : " << mol.filename << "\n";
    //    std::cout << "  Comment Line   : " << mol.commentLine << "\n";
        std::cout << "  Atom Count     : " << mol.numAtoms << "\n";

        // Prints SRD and SRV if computed
        m_faissManager.printDescriptors(index);

        std::cout << "-------------------------------------------------------\n";
        std::cout << "  Atomic Coordinates:\n";

        size_t printLimit = std::min(mol.numAtoms, (size_t)40);
        for (size_t i = 0; i < printLimit; ++i) {
            const auto& a = mol.atoms[i];
            std::cout << "    [" << std::setw(3) << i << "] " 
                      << std::setw(3) << a.symbol << " | X: "
                      << std::setw(11) << std::fixed << std::setprecision(6) << a.x << ", Y: "
                      << std::setw(11) << a.y << ", Z: "
                      << std::setw(11) << a.z << " | Q: "
                      << std::setw(8) << a.charge << "\n";
        }
        if (mol.numAtoms > printLimit) {
            std::cout << "    ... (showing first " << printLimit << " of " << mol.numAtoms << " atoms)\n";
        }
        std::cout << "=======================================================\n\n";
    }

    bool setMatchVisible(size_t a, size_t b, bool v) {
    if (a >= m_entries.size() || b >= m_entries.size()) { std::cout << "[Matching] Index out of range.\n"; return false; }
    auto* m = MatchingViz::find(m_entries[a].id, m_entries[b].id);
    if (!m) { std::cout << "[Matching] No stored matching for this pair.\n"; return false; }
    m->visible = v;
    return true;
}

bool removeMatch(size_t a, size_t b) {
    if (a >= m_entries.size() || b >= m_entries.size()) { std::cout << "[Matching] Index out of range.\n"; return false; }
    return MatchingViz::remove(m_entries[a].id, m_entries[b].id);
}

bool applyAlignment(size_t a, size_t b) {
    if (a >= m_entries.size() || b >= m_entries.size()) { std::cout << "[Matching] Index out of range.\n"; return false; }
    return MatchingViz::applyAlignment(m_entries[a].id, m_entries[b].id);
}

    size_t size() const { return m_entries.size(); }
    void clearAll() { 
        MatchingViz::clearAll();
        m_entries.clear(); 
        m_faissManager.clear();
    }

private:
    std::vector<MoleculeEntry> m_entries;
    size_t m_nextId = 0;
    FaissIndexManager m_faissManager;

    void printProgressBar(size_t current, size_t total) const {
        int barWidth = 40;
        float progress = static_cast<float>(current) / total;
        int tensorWidth = static_cast<int>(barWidth * progress);

        std::cout << "\r[";
        for (int i = 0; i < barWidth; ++i) {
            if (i < tensorWidth) std::cout << "=";
            else if (i == tensorWidth) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0f) << "% (" << current << "/" << total << ")" << std::flush;
    }
};