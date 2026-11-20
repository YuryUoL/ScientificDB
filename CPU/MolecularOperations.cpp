#include "MolecularOperations.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "Visualization/PolyscopeBridge.h"
#include "Visualization/MolecularGraphViz.h"

MolecularOperations::MolecularOperations(MolecularDatasetStore &store)
    : m_store(store)
{
}

void MolecularOperations::ensureDescriptorsUpToDate()
{
    const auto &entries = m_store.molecules();
    auto &faiss = m_store.faiss();

    if (!m_store.descriptorsDirty() && faiss.descriptorCount() == entries.size())
        return;

    faiss.syncWithEntries(
        entries.size(),
        [&entries](size_t i) -> const std::vector<AtomRecord> &
        { return entries[i].atoms; });
    m_store.setDescriptorsDirty(false);
}

void MolecularOperations::computeDescriptors()
{
    const auto &entries = m_store.molecules();
    m_store.faiss().computeDescriptorsForAll(
        entries.size(),
        [&entries](size_t i) -> const std::vector<AtomRecord> &
        { return entries[i].atoms; });
    m_store.setDescriptorsDirty(false);
}

void MolecularOperations::flushDescriptors()
{
    m_store.faiss().clear();
    m_store.setDescriptorsDirty(true);
}

bool MolecularOperations::searchKNN(size_t index, const std::string &kStr, const std::string &metricType, const std::string &csvPath, bool restrictToLast)
{
    ensureDescriptorsUpToDate();
    return m_store.faiss().searchKNN(index, kStr, metricType, csvPath, restrictToLast);
}

bool MolecularOperations::searchKNNForAll(const std::string &kStr, const std::string &metricType, const std::string &csvPath)
{
    ensureDescriptorsUpToDate();
    return m_store.faiss().searchKNNForAll(kStr, metricType, csvPath);
}

float MolecularOperations::computePairMetric(size_t idxA, size_t idxB, const std::string &metricType)
{
    ensureDescriptorsUpToDate();
    return m_store.faiss().computePairDistance(idxA, idxB, metricType);
}

void MolecularOperations::printDistanceDistribution() const
{
    m_store.faiss().printDistanceDistribution();
}

std::vector<float> MolecularOperations::getSDV(size_t index) const
{
    const auto *desc = m_store.faiss().descriptor(index);
    if (!desc || !desc->computed)
        return {};
    return desc->sdv;
}

std::vector<float> MolecularOperations::getSRV(size_t index) const
{
    const auto *desc = m_store.faiss().descriptor(index);
    if (!desc || !desc->computed)
        return {};
    return desc->srv;
}

std::vector<float> MolecularOperations::getPDD(size_t index) const
{
    const auto *desc = m_store.faiss().descriptor(index);
    if (!desc || !desc->computed)
        return {};
    return desc->pdd;
}

void MolecularOperations::writeMatchesCsv(const std::string &csvPath, const std::vector<DupMatch> &matches)
{
    if (csvPath.empty() || matches.empty())
        return;

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

std::vector<DupMatch> MolecularOperations::runDuplicateCheck(size_t queryIdx, const std::string &csvPath, const DupCheckParams &p, DupRunStats &st)
{
    const auto &entries = m_store.molecules();
    DuplicateChecker checker;
    if (queryIdx >= entries.size())
    {
        std::cout << "[XyzStore] Error: Query index out of range.\n";
        st = DupRunStats{};
        return {};
    }

    ensureDescriptorsUpToDate();

    const auto &qMol = entries[queryIdx];
    const auto *qDesc = m_store.faiss().descriptor(queryIdx);
    if (!qDesc || !qDesc->computed)
    {
        std::cout << "[XyzStore] Error: Descriptors could not be computed.\n";
        st = DupRunStats{};
        return {};
    }

    std::vector<DupMatch> matches = checker.run(
        m_store.faiss(),
        queryIdx,
        *qDesc,
        qMol.atoms,
        [&entries](size_t i) -> const std::vector<AtomRecord> &
        { return entries[i].atoms; },
        [&entries](size_t i) -> std::string
        { return entries[i].commentLine; },
        p,
        st);

    for (const auto &m : matches)
    {
        const auto &molA = entries[m.queryIdx];
        const auto &molB = entries[m.matchIdx];
        MatchingViz::store(molA.id, molB.id, molA.filename, molB.filename, m.align);
    }

    writeMatchesCsv(csvPath, matches);
    return matches;
}

std::vector<DupMatch> MolecularOperations::runBatchDuplicateCheck(const std::string &csvPath, const DupCheckParams &p, DupRunStats &st)
{
    const auto &entries = m_store.molecules();
    DuplicateChecker checker;
    size_t totalMols = entries.size();
    if (totalMols < 2)
    {
        std::cout << "[XyzStore] Error: Not enough molecules loaded for batch duplicate check.\n";
        st = DupRunStats{};
        return {};
    }

    ensureDescriptorsUpToDate();

    std::vector<DupMatch> matches = checker.runAll(
        m_store.faiss(),
        totalMols,
        [&entries](size_t i) -> const std::vector<AtomRecord> &
        { return entries[i].atoms; },
        [&entries](size_t i) -> std::string
        { return entries[i].commentLine; },
        p,
        st);

    for (const auto &m : matches)
    {
        const auto &molA = entries[m.queryIdx];
        const auto &molB = entries[m.matchIdx];
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

    m_store.lastBatchMatches() = matches;
    return matches;
}

bool MolecularOperations::printPairAlignmentWithStats(size_t idxA, size_t idxB, AlignmentOptions opt)
{
    const auto &entries = m_store.molecules();
    if (idxA >= entries.size() || idxB >= entries.size())
    {
        std::cout << "[XyzStore] Error: Index out of range.\n";
        return false;
    }

    const auto &molA = entries[idxA];
    const auto &molB = entries[idxB];

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

AlignmentResult MolecularOperations::computePairAlignmentWithStats(size_t idxA, size_t idxB, AlignmentOptions opt)
{
    const auto &entries = m_store.molecules();
    if (idxA >= entries.size() || idxB >= entries.size())
    {
        return AlignmentResult{};
    }

    const auto &molA = entries[idxA];
    const auto &molB = entries[idxB];

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

bool MolecularOperations::setMatchVisible(size_t a, size_t b, bool v)
{
    const auto &entries = m_store.molecules();
    if (a >= entries.size() || b >= entries.size())
    {
        std::cout << "[Matching] Index out of range.\n";
        return false;
    }
    auto *m = MatchingViz::find(entries[a].id, entries[b].id);
    if (!m)
    {
        std::cout << "[Matching] No stored matching for this pair.\n";
        return false;
    }
    m->visible = v;
    return true;
}

void MolecularOperations::setAllMatchesVisible(bool v)
{
    MatchingViz::setAllVisible(v);
}

bool MolecularOperations::removeMatch(size_t a, size_t b)
{
    const auto &entries = m_store.molecules();
    if (a >= entries.size() || b >= entries.size())
    {
        std::cout << "[Matching] Index out of range.\n";
        return false;
    }
    return MatchingViz::remove(entries[a].id, entries[b].id);
}

bool MolecularOperations::applyAlignment(size_t a, size_t b)
{
    const auto &entries = m_store.molecules();
    if (a >= entries.size() || b >= entries.size())
    {
        std::cout << "[Matching] Index out of range.\n";
        return false;
    }
    return MatchingViz::applyAlignment(entries[a].id, entries[b].id);
}

bool MolecularOperations::removeByIndex(size_t index)
{
    auto &entries = m_store.molecules();
    if (index < entries.size())
    {
        MatchingViz::removeForMolecule(entries[index].id);
        size_t id = entries[index].id;
        entries.erase(entries.begin() + index);
        m_store.faiss().removeEntryAt(index);
        m_store.setDescriptorsDirty(m_store.faiss().descriptorCount() != entries.size());
        std::cout << "[XyzStore] Removed molecule at index " << index << " (ID: " << id << ")\n";
        return true;
    }
    std::cout << "[XyzStore] Error: Index " << index << " out of range.\n";
    return false;
}

bool MolecularOperations::visualizeIndex(size_t index) const
{
    const auto &entries = m_store.molecules();
    if (index >= entries.size())
    {
        std::cout << "[XyzStore] Error: Index " << index << " out of range (Total: " << entries.size() << ").\n";
        return false;
    }

    const auto &mol = entries[index];
    PolyscopeBridge::visualizeMolecule(mol.id, mol.filename, mol.atoms);
    std::cout << "[Polyscope] Successfully rendered molecule index " << index << " (" << mol.filename << ") as point cloud.\n";
    return true;
}

bool MolecularOperations::unviewIndex(size_t index) const
{
    const auto &entries = m_store.molecules();
    if (index >= entries.size())
    {
        std::cout << "[XyzStore] Error: Index " << index << " out of range (Total: " << entries.size() << ").\n";
        return false;
    }

    const auto &mol = entries[index];
    PolyscopeBridge::removeVisualization(mol.id, mol.filename);
    return true;
}

void MolecularOperations::exploreClusters()
{
    const auto &entries = m_store.molecules();
    const auto &batchMatches = m_store.lastBatchMatches();

    if (batchMatches.empty())
    {
        std::cout << "[XyzStore] Error: No recent batch duplicate check found. Run 'dup_check all' first.\n";
        return;
    }

    std::vector<size_t> atomCounts(entries.size());
    for (size_t i = 0; i < entries.size(); ++i)
    {
        atomCounts[i] = entries[i].numAtoms;
    }

    MolecularGraphViz::buildClusters(batchMatches, atomCounts);
}

void MolecularOperations::setGraphVisible(bool visible) const
{
    MolecularGraphViz::setGraphVisible(visible);
}

void MolecularOperations::clearGraph() const
{
    MolecularGraphViz::clearGraph();
}