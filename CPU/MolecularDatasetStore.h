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
#include "Duplicatechecker.h"
#include "MolecularGraphViz.h"

struct MoleculeEntry
{
    size_t id;
    std::string filename;
    std::string commentLine;
    std::vector<AtomRecord> atoms;
    size_t numAtoms = 0;
};

class MolecularDatasetStore
{
public:
    MolecularDatasetStore() = default;
    ~MolecularDatasetStore() { clearAll(); }

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
        return true;
    }

    // Parses a raw XYZ string content directly into a molecule entry in RAM
    bool addStringContent(const std::string& filename, const std::string& content)
    {
        std::istringstream file(content);
        std::string line;
        if (!std::getline(file, line))
            return false;

        int numAtoms = 0;
        try {
            numAtoms = std::stoi(line);
        } catch (...) {
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

    // Allows direct insertion of programmatically generated molecules
    size_t addSyntheticEntry(const std::string& filename, const std::string& comment, const std::vector<AtomRecord>& atoms) {
        MoleculeEntry mol;
        mol.id = m_nextId++;
        mol.filename = filename;
        mol.commentLine = comment;
        mol.atoms = atoms;
        mol.numAtoms = atoms.size();
        m_entries.push_back(std::move(mol));
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
        const auto &qMol = m_entries[queryIdx];
        const auto *qDesc = m_faissManager.descriptor(queryIdx);
        if (!qDesc || !qDesc->computed)
        {
            std::cout << "[XyzStore] Error: Descriptors not computed. Run 'compute' first.\n";
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

    bool searchKNN(size_t index, const std::string &kStr, const std::string &metricType, const std::string &csvPath = "", bool restrictToLast = false) const
    {
        return m_faissManager.searchKNN(index, kStr, metricType, csvPath, restrictToLast);
    }

    void printDistanceDistribution() const
    {
        m_faissManager.printDistanceDistribution();
    }

    void printPairAlignment(size_t idxA, size_t idxB, AlignmentOptions opt = AlignmentOptions()) const
    {
        if (idxA >= m_entries.size() || idxB >= m_entries.size())
        {
            std::cout << "[XyzStore] Error: Index out of range.\n";
            return;
        }

        const auto &molA = m_entries[idxA];
        const auto &molB = m_entries[idxB];

        if (molA.numAtoms != molB.numAtoms || molA.numAtoms == 0)
        {
            std::cout << "[XyzStore] Error: Molecules must have an equal and non-zero number of atoms for alignment.\n";
            return;
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
            return;
        }

        MatchingViz::store(molA.id, molB.id, molA.filename, molB.filename, res);
        std::cout << "[Matching] Stored (hidden). Use 'show_match <a> <b>' to display it.\n";
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
            std::cout << "[XyzStore] Removed molecule at index " << index << " (ID: " << id << ")\n";
            return true;
        }
        std::cout << "[XyzStore] Error: Index " << index << " out of range.\n";
        return false;
    }

    bool searchKNNForAll(const std::string &kStr, const std::string &metricType, const std::string &csvPath) const
    {
        return m_faissManager.searchKNNForAll(kStr, metricType, csvPath);
    }

    float computePairMetric(size_t idxA, size_t idxB, const std::string &metricType) const
    {
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
    }

    void flushDescriptors()
    {
        m_faissManager.clear();
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
        std::cout << "  Atom Count     : " << mol.numAtoms << "\n";

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
        MatchingViz::clearAll();
        m_entries.clear();
        m_faissManager.clear();
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
    size_t m_nextId = 0;
    FaissIndexManager m_faissManager;
    std::vector<DupMatch> m_lastBatchMatches;

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