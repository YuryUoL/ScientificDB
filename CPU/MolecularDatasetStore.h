#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include "Visualization/PolyscopeBridge.h"
#include "FaissIndexManager.h"
#include "Duplicatechecker.h"
#include "Visualization/MolecularGraphViz.h"
#include "AtomicClassifier.h"
#include "Visualization/Matchingvisualizer.h"
#include "Visualization/AtomicMatchingViz.h"

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

    // --- Atomic Descriptors ---
    std::vector<float> sortedDistances;    // N-1 distances sorted ascending (excluding center atom at 0)
    std::vector<size_t> sortedAtomIndices; // Original atom indices sorted by distance to centroid
    bool descriptorsComputed = false;
};

class MolecularDatasetStore
{
public:
    MolecularDatasetStore() = default;
    ~MolecularDatasetStore() { clearAll(); }

    // --- Basic Dataset Loading & Lifecycle ---
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

    void clearAll()
{
    PolyscopeBridge::clearAll();
    clearAtomicCloudsStorage();
    MatchingViz::clearAll();
    m_entries.clear();
    m_faissManager.clear();
    m_lastBatchMatches.clear();
    m_descriptorsDirty = false;

    // RESET THE MOLECULE ID COUNTER BACK TO 0!
    m_nextId = 0;
}

    void clearAtomicCloudsStorage()
    {
        for (size_t i = 0; i < m_atomicEntries.size(); ++i)
        {
            const auto &ac = m_atomicEntries[i];
            std::string cloudName = AtomicMatchingViz::atomicCloudName(i, ac.parentMoleculeIdx, ac.symbol, ac.atomIndex);
            if (polyscope::hasPointCloud(cloudName))
            {
                polyscope::removePointCloud(cloudName, true);
            }
        }
        m_atomicEntries.clear();
        m_nextAtomicId = 0;
    }

    // --- Basic Molecule Metadata & Inspection ---
    size_t size() const { return m_entries.size(); }

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

        size_t printLimit = std::min(mol.atoms.size(), size_t(100));
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

    // --- Direct Accessors for MolecularOperations & AtomicOperations ---
    std::vector<MoleculeEntry> &molecules() { return m_entries; }
    const std::vector<MoleculeEntry> &molecules() const { return m_entries; }

    std::vector<AtomicCloudEntry> &atomicClouds() { return m_atomicEntries; }
    const std::vector<AtomicCloudEntry> &atomicClouds() const { return m_atomicEntries; }

    FaissIndexManager &faiss() { return m_faissManager; }
    const FaissIndexManager &faiss() const { return m_faissManager; }

    std::vector<DupMatch> &lastBatchMatches() { return m_lastBatchMatches; }
    const std::vector<DupMatch> &lastBatchMatches() const { return m_lastBatchMatches; }

    bool descriptorsDirty() const { return m_descriptorsDirty; }
    void setDescriptorsDirty(bool dirty) { m_descriptorsDirty = dirty; }

    size_t nextAtomicId() const { return m_nextAtomicId; }
    size_t consumeNextAtomicId() { return m_nextAtomicId++; }
    void setNextAtomicId(size_t id) { m_nextAtomicId = id; }

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

private:
    std::vector<MoleculeEntry> m_entries;
    std::vector<AtomicCloudEntry> m_atomicEntries;
    size_t m_nextId = 0;
    size_t m_nextAtomicId = 0;
    FaissIndexManager m_faissManager;
    std::vector<DupMatch> m_lastBatchMatches;
    bool m_descriptorsDirty = false;
};