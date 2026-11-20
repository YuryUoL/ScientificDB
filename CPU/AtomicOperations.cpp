#include "AtomicOperations.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <cmath>
#include <limits>
#include <algorithm>
#include <omp.h>
#include <oneapi/tbb/parallel_sort.h>
#include "Visualization/PolyscopeBridge.h"

AtomicOperations::AtomicOperations(MolecularDatasetStore &store)
    : m_store(store)
{
}

void AtomicOperations::generateAllAtomicClouds()
{
    const auto &entries = m_store.molecules();
    auto &atomicEntries = m_store.atomicClouds();

    atomicEntries.clear();
    size_t totalMols = entries.size();
    if (totalMols == 0)
    {
        std::cout << "[AtomicStore] No molecules loaded to expand.\n";
        return;
    }

    std::cout << "[AtomicStore] Expanding " << totalMols << " molecules into atomic neighborhoods in parallel...\n";
    auto tStart = std::chrono::high_resolution_clock::now();

    int maxThreads = omp_get_max_threads();
    std::vector<std::vector<AtomicCloudEntry>> threadPoolEntries(maxThreads);
    std::atomic<size_t> completedMols{0};

#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        size_t localNextId = static_cast<size_t>(tid) * 10000000ULL;

#pragma omp for schedule(dynamic, 64)
        for (long long idx = 0; idx < static_cast<long long>(totalMols); ++idx)
        {
            size_t molIdx = static_cast<size_t>(idx);
            const auto &mol = entries[molIdx];
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
                    m_store.printProgressBar(done, totalMols);
                }
            }
        }
    }

    for (int tid = 0; tid < maxThreads; ++tid)
    {
        for (auto &ac : threadPoolEntries[tid])
        {
            atomicEntries.push_back(std::move(ac));
        }
    }

    tbb::parallel_sort(atomicEntries.begin(), atomicEntries.end(), [](const auto &a, const auto &b)
                       {
        if (a.parentMoleculeIdx != b.parentMoleculeIdx)
            return a.parentMoleculeIdx < b.parentMoleculeIdx;
        return a.atomIndex < b.atomIndex; });

    for (size_t i = 0; i < atomicEntries.size(); ++i)
    {
        atomicEntries[i].id = i;
    }
    m_store.setNextAtomicId(atomicEntries.size());

    auto tEnd = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    std::cout << "\n[AtomicStore] Successfully generated and sorted " << atomicEntries.size()
              << " atomic clouds across " << totalMols << " molecules in "
              << std::fixed << std::setprecision(2) << elapsedMs << " ms.\n";
}

void AtomicOperations::generateAtomicCloudsFor(size_t molIdx)
{
    const auto &entries = m_store.molecules();
    auto &atomicEntries = m_store.atomicClouds();

    if (molIdx >= entries.size())
    {
        std::cout << "[AtomicStore] Error: Molecule index " << molIdx << " out of range.\n";
        return;
    }
    const auto &mol = entries[molIdx];

    for (size_t i = 0; i < mol.atoms.size(); ++i)
    {
        const auto &centerAtom = mol.atoms[i];

        AtomicCloudEntry atomicCloud;
        atomicCloud.id = m_store.consumeNextAtomicId();
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
        atomicEntries.push_back(std::move(atomicCloud));
    }
}

void AtomicOperations::clearAtomicClouds()
{
    m_store.clearAtomicCloudsStorage();
    std::cout << "[AtomicStore] Cleared all atomic neighborhood clouds.\n";
}

size_t AtomicOperations::atomicSize() const
{
    return m_store.atomicClouds().size();
}

std::string AtomicOperations::getAtomicName(size_t index) const
{
    const auto &atomicEntries = m_store.atomicClouds();
    if (index >= atomicEntries.size())
        return "";
    return atomicEntries[index].name;
}

size_t AtomicOperations::getAtomicAtomCount(size_t index) const
{
    const auto &atomicEntries = m_store.atomicClouds();
    if (index >= atomicEntries.size())
        return 0;
    return atomicEntries[index].numAtoms;
}

void AtomicOperations::computeAtomicDescriptors()
{
    auto &atomicEntries = m_store.atomicClouds();
    if (atomicEntries.empty())
    {
        std::cout << "[AtomicStore] Error: No atomic clouds generated. Run 'expand_atoms' first.\n";
        return;
    }

    size_t totalClouds = atomicEntries.size();
    std::cout << "[AtomicStore] Computing atomic descriptors for " << totalClouds << " atomic clouds in parallel...\n";
    auto tStart = std::chrono::high_resolution_clock::now();

    std::atomic<size_t> completedClouds{0};

#pragma omp parallel for schedule(dynamic, 64)
    for (long long idx = 0; idx < static_cast<long long>(totalClouds); ++idx)
    {
        auto &ac = atomicEntries[static_cast<size_t>(idx)];

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

            if (dist < 1e-6f)
            {
                continue;
            }
            pairs.push_back({j, dist});
        }

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
                m_store.printProgressBar(done, totalClouds);
            }
        }
    }

    auto tEnd = std::chrono::high_resolution_clock::now();
    double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    std::cout << "\n[AtomicStore] Successfully computed atomic descriptors in "
              << std::fixed << std::setprecision(2) << elapsedMs << " ms.\n";
}

float AtomicOperations::computeAtomicPairDistance(size_t idxA, size_t idxB, const std::string &metric, size_t N) const
{
    const auto &atomicEntries = m_store.atomicClouds();
    if (idxA >= atomicEntries.size() || idxB >= atomicEntries.size())
    {
        std::cout << "[AtomicStore] Error: Atomic cloud index out of range.\n";
        return -1.0f;
    }

    const auto &acA = atomicEntries[idxA];
    const auto &acB = atomicEntries[idxB];

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

bool AtomicOperations::computeAtomicAlignerDistance(size_t idxA, size_t idxB, size_t N, const AlignmentOptions &opt)
{
    const auto &atomicEntries = m_store.atomicClouds();
    if (idxA >= atomicEntries.size() || idxB >= atomicEntries.size())
    {
        std::cout << "[AtomicStore] Error: Index out of range.\n";
        return false;
    }

    const auto &acA = atomicEntries[idxA];
    const auto &acB = atomicEntries[idxB];

    if (!acA.descriptorsComputed || !acB.descriptorsComputed)
    {
        std::cout << "[AtomicStore] Error: Descriptors not computed. Run 'ComputeAtomicDescriptors' first.\n";
        return false;
    }

    size_t limitA = (N > 0) ? std::min(N, acA.sortedAtomIndices.size()) : acA.sortedAtomIndices.size();
    if (limitA == 0)
    {
        std::cout << "[AtomicStore] Error: Cloud A subset is empty (N=0 or no neighbor atoms).\n";
        return false;
    }

    std::vector<size_t> idxB_subset = acB.sortedAtomIndices;
    if (idxB_subset.empty())
    {
        std::cout << "[AtomicStore] Error: Cloud B subset is empty (no neighbor atoms).\n";
        return false;
    }

    if (idxB_subset.size() < limitA)
    {
        std::cout << "[AtomicStore] Error: Cloud B subset size (" << idxB_subset.size()
                  << ") is smaller than Cloud A subset size (" << limitA
                  << "). Clamping subset size to " << idxB_subset.size() << ".\n";
        limitA = idxB_subset.size();
    }

    std::vector<size_t> idxA_subset(acA.sortedAtomIndices.begin(),
                                    acA.sortedAtomIndices.begin() + limitA);

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

    const size_t m = idxA_subset.size();
    const size_t n = idxB_subset.size();
    float bottleneckDist = res.bestBottleneck;
    if (m <= AtomicBottleneckEvaluator::MAX_VERTICES && n <= AtomicBottleneckEvaluator::MAX_VERTICES)
    {
        const float *R = res.rotation;
        std::vector<float> cost(m * n, std::numeric_limits<float>::infinity());
        float lo = 0.0f;
        float hi = 0.0f;

        for (size_t i = 0; i < m; ++i)
        {
            const auto &pA = cloudA[idxA_subset[i]];
            float ax = R[0] * pA.x + R[1] * pA.y + R[2] * pA.z;
            float ay = R[3] * pA.x + R[4] * pA.y + R[5] * pA.z;
            float az = R[6] * pA.x + R[7] * pA.y + R[8] * pA.z;

            float rowMin = std::numeric_limits<float>::infinity();
            float rowMax = 0.0f;
            for (size_t j = 0; j < n; ++j)
            {
                if (localOpt.requireSameSymbol && symbolsA[idxA_subset[i]] != symbolsB[idxB_subset[j]])
                    continue;

                const auto &pB = cloudB[idxB_subset[j]];
                float dx = ax - pB.x;
                float dy = ay - pB.y;
                float dz = az - pB.z;
                float d = std::sqrt(dx * dx + dy * dy + dz * dz);
                cost[i * n + j] = d;
                if (d < rowMin)
                    rowMin = d;
                if (d > rowMax)
                    rowMax = d;
            }
            if (rowMin < std::numeric_limits<float>::infinity() && rowMin > lo)
                lo = rowMin;
            if (rowMax > hi)
                hi = rowMax;
        }
        if (lo <= hi && hi > 0.0f)
        {
            bottleneckDist = AtomicBottleneckEvaluator::bottleneckInjective(cost.data(), m, n, lo, hi);
        }
    }
    else if (n <= AtomicBottleneckEvaluator::MAX_VERTICES)
    {
        std::vector<float> dA(acA.sortedDistances.begin(), acA.sortedDistances.begin() + m);
        std::vector<float> dB = acB.sortedDistances;
        std::sort(dA.begin(), dA.end());
        std::sort(dB.begin(), dB.end());
        bottleneckDist = AtomicBottleneckEvaluator::injectiveBottleneck1D(dA.data(), m, dB.data(), n);
    }

    AtomicMatchingViz::store(
        idxA, idxB,
        acA.parentMoleculeIdx, acA.symbol, acA.atomIndex,
        acB.parentMoleculeIdx, acB.symbol, acB.atomIndex,
        res, idxA_subset, idxB_subset);

    std::cout << "\n=======================================================\n";
    std::cout << "  ATOMIC CLOUD ALIGNMENT & MATCHING (aligner - NO CENTER ATOMS)\n";
    std::cout << "=======================================================\n";
    std::cout << "  Atomic Cloud A   : " << idxA << " (" << acA.name << ")\n";
    std::cout << "  Atomic Cloud B   : " << idxB << " (" << acB.name << ")\n";
    std::cout << "  Subset Size A    : " << idxA_subset.size() << " neighbor atoms (Center excluded)\n";
    std::cout << "  Subset Size B    : " << idxB_subset.size() << " neighbor atoms (Center excluded)\n";
    std::cout << "  Bottleneck Dist  : " << std::fixed << std::setprecision(6) << bottleneckDist << "\n";
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

std::vector<AtomicClassifyMatch> AtomicOperations::runBatchAtomicClassify(const std::string &csvPath,
                                                                        const AtomicClassifyParams &p,
                                                                        AtomicClassifyStats &st)
{
    auto &atomicEntries = m_store.atomicClouds();
    if (atomicEntries.empty())
    {
        generateAllAtomicClouds();
    }
    bool needDescriptors = false;
    for (const auto &ac : atomicEntries)
    {
        if (!ac.descriptorsComputed)
        {
            needDescriptors = true;
            break;
        }
    }
    if (needDescriptors)
    {
        computeAtomicDescriptors();
    }

    AtomicClassifier classifier;
    std::vector<AtomicClassifyMatch> matches = classifier.classifyAll(atomicEntries, p, st);

    for (const auto &m : matches)
    {
        const auto &acA = atomicEntries[m.atomicIdxA];
        const auto &acB = atomicEntries[m.atomicIdxB];

        size_t limitA = (p.N > 0) ? std::min(p.N, acA.sortedAtomIndices.size()) : acA.sortedAtomIndices.size();
        if (acB.sortedAtomIndices.size() < limitA)
            limitA = acB.sortedAtomIndices.size();

        std::vector<size_t> idxA_subset(acA.sortedAtomIndices.begin(),
                                        acA.sortedAtomIndices.begin() + limitA);

        AtomicMatchingViz::store(
            m.atomicIdxA, m.atomicIdxB,
            acA.parentMoleculeIdx, acA.symbol, acA.atomIndex,
            acB.parentMoleculeIdx, acB.symbol, acB.atomIndex,
            m.alignAB, idxA_subset, acB.sortedAtomIndices);

        size_t limitB = (p.N > 0) ? std::min(p.N, acB.sortedAtomIndices.size()) : acB.sortedAtomIndices.size();
        if (acA.sortedAtomIndices.size() < limitB)
            limitB = acA.sortedAtomIndices.size();

        std::vector<size_t> idxB_subset(acB.sortedAtomIndices.begin(),
                                        acB.sortedAtomIndices.begin() + limitB);

        AtomicMatchingViz::store(
            m.atomicIdxB, m.atomicIdxA,
            acB.parentMoleculeIdx, acB.symbol, acB.atomIndex,
            acA.parentMoleculeIdx, acA.symbol, acA.atomIndex,
            m.alignBA, idxB_subset, acA.sortedAtomIndices);
    }

    if (!csvPath.empty())
    {
        AtomicClassifier::writeCsv(csvPath, matches);
    }

    return matches;
}

bool AtomicOperations::visualizeAtomicIndex(size_t index) const
{
    const auto &atomicEntries = m_store.atomicClouds();
    if (index >= atomicEntries.size())
    {
        std::cout << "[AtomicStore] Error: Atomic index " << index << " out of range.\n";
        return false;
    }
    const auto &ac = atomicEntries[index];
    std::string cloudName = AtomicMatchingViz::atomicCloudName(index, ac.parentMoleculeIdx, ac.symbol, ac.atomIndex);

    PolyscopeBridge::visualizeByName(cloudName, ac.atoms, false);

    std::cout << "[Polyscope] Rendered atomic neighborhood (centered on atom): " << cloudName << "\n";
    return true;
}

bool AtomicOperations::unviewAtomicIndex(size_t index) const
{
    const auto &atomicEntries = m_store.atomicClouds();
    if (index >= atomicEntries.size())
        return false;
    const auto &ac = atomicEntries[index];
    std::string cloudName = AtomicMatchingViz::atomicCloudName(index, ac.parentMoleculeIdx, ac.symbol, ac.atomIndex);
    PolyscopeBridge::removeByName(cloudName);
    return true;
}

AtomicExpressResult AtomicOperations::runAtomicExpressCount(const AtomicExpressParams& p,
                                                            bool verifyAgainstBruteForce)
{
    auto& atomicEntries = m_store.atomicClouds();

    // --- lazy bootstrap ---
    if (atomicEntries.empty())
    {
        std::cout << "[AtomicExpress] No atomic clouds present, generating them first...\n";
        generateAllAtomicClouds();
    }
    bool needDescriptors = false;
    for (const auto& ac : atomicEntries)
        if (!ac.descriptorsComputed) { needDescriptors = true; break; }
    if (needDescriptors)
    {
        std::cout << "[AtomicExpress] Descriptors missing, computing them first...\n";
        computeAtomicDescriptors();
    }

    if (atomicEntries.size() < 2)
    {
        std::cout << "[AtomicExpress] Need at least 2 atomic clouds to count pairs.\n";
        return {};
    }

    // --- banner ---
    const float rSrv   = p.srvFactor   * p.epsilon * (1.0f + p.relTol) + 1e-6f;
    const float rAlign = p.alignFactor * p.epsilon * (1.0f + p.relTol) + 1e-6f;

    std::cout << "\n=======================================================\n"
              << "  LAUNCHING ATOMIC EXPRESS ("
              << (p.runAligner ? "Stage-1 SRV + Stage-2 Aligner" : "Stage-1 SRV only")
              << ")\n"
              << "=======================================================\n"
              << "  Epsilon (epsilon)   : " << std::fixed << std::setprecision(4) << p.epsilon << " A\n"
              << "  Neighbors (N)       : " << (p.N == 0 ? "ALL neighbors" : std::to_string(p.N)) << "\n"
              << "  Consider Same Type  : " << (p.considerSame ? "true (all pairs)" : "false (different center types only)") << "\n"
              << "  SRV Factor (srvf)   : " << std::fixed << std::setprecision(2) << p.srvFactor
              << "x (limit: " << rSrv << " A)\n"
              << "  Slack Tol (tol)     : " << p.relTol << "\n"
              << "  Grid Dim (grid)     : " << p.gridDim << " (capped at 4)\n"
              << "  Execution Mode      : " << (p.parallel ? "Parallel (OpenMP)" : "Single-threaded")
              << " (threads: " << (p.numThreads > 0 ? std::to_string(p.numThreads) : "auto/max") << ")\n"
              << "-------------------------------------------------------\n"
              << "  CSV Export          : " << (p.csvPath.empty() ? "(disabled)" : p.csvPath) << "\n"
              << "  CSV Cloud Names     : " << (p.csvIncludeCloudNames ? "on" : "off") << "\n"
              << "  CSV Atom Counts     : " << (p.csvIncludeAtomCounts ? "on" : "off") << "\n"
              << "-------------------------------------------------------\n"
              << "  Run Aligner (Stage2): " << (p.runAligner ? "true" : "false") << "\n";
    if (p.runAligner)
    {
        std::cout << "  Aligner Optimal     : " << (p.optimal ? "true" : "false") << "\n"
                  << "  Align Factor        : " << std::fixed << std::setprecision(2) << p.alignFactor
                  << "x (limit: " << rAlign << " A)\n"
                  << "  Species Constraint  : " << (p.requireSameNeighborSpecies ? "true" : "false") << "\n"
                  << "  Kabsch Refinement   : " << p.refineIterations << " rounds\n"
                  << "  Max Matches Stored  : " << (p.maxMatches == 0 ? "ALL" : std::to_string(p.maxMatches)) << "\n";
    }
    std::cout << "  Verify vs O(M^2)    : " << (verifyAgainstBruteForce ? "true" : "false") << "\n"
              << "=======================================================\n\n"
              << std::flush;

    // --- run express ---
    AtomicClassifierExpress express;
    AtomicExpressResult R = express.countCrossLabelPairs(atomicEntries, p);
    AtomicClassifierExpress::printReport(p, R);

    // --- push aligned pairs into the visualizer (both directions) ---
    // Mirrors runBatchAtomicClassify: only aligned pairs, two calls per pair.
    if (p.runAligner && !R.pairs.empty())
    {
        size_t stored = 0;
        for (const auto& rec : R.pairs)
        {
            if (!rec.aligned) continue;
            const auto& acA = atomicEntries[rec.idxA];
            const auto& acB = atomicEntries[rec.idxB];

            // ---- A -> B ----
            size_t limitA = (p.N > 0)
                                ? std::min(p.N, acA.sortedAtomIndices.size())
                                : acA.sortedAtomIndices.size();
            if (acB.sortedAtomIndices.size() < limitA)
                limitA = acB.sortedAtomIndices.size();
            std::vector<size_t> idxA_subset(acA.sortedAtomIndices.begin(),
                                            acA.sortedAtomIndices.begin() + limitA);

            AtomicMatchingViz::store(
                rec.idxA, rec.idxB,
                acA.parentMoleculeIdx, acA.symbol, acA.atomIndex,
                acB.parentMoleculeIdx, acB.symbol, acB.atomIndex,
                rec.alignAB, idxA_subset, acB.sortedAtomIndices);

            // ---- B -> A ----
            size_t limitB = (p.N > 0)
                                ? std::min(p.N, acB.sortedAtomIndices.size())
                                : acB.sortedAtomIndices.size();
            if (acA.sortedAtomIndices.size() < limitB)
                limitB = acA.sortedAtomIndices.size();
            std::vector<size_t> idxB_subset(acB.sortedAtomIndices.begin(),
                                            acB.sortedAtomIndices.begin() + limitB);

            AtomicMatchingViz::store(
                rec.idxB, rec.idxA,
                acB.parentMoleculeIdx, acB.symbol, acB.atomIndex,
                acA.parentMoleculeIdx, acA.symbol, acA.atomIndex,
                rec.alignBA, idxB_subset, acA.sortedAtomIndices);

            ++stored;
        }
        std::cout << "[AtomicExpress] Stored " << stored
                  << " aligned pairs in AtomicMatchingViz (both directions).\n";
    }

    // --- optional CSV ---
    if (!p.csvPath.empty())
        AtomicClassifierExpress::writeCsv(p.csvPath, R);

    // --- optional correctness oracle ---
    if (verifyAgainstBruteForce)
    {
        std::cout << "[AtomicExpress] Running O(M^2) brute-force oracle for verification...\n";
        auto t0 = std::chrono::high_resolution_clock::now();
        AtomicExpressResult B = AtomicClassifierExpress::countBruteForce(atomicEntries, p);
        auto t1 = std::chrono::high_resolution_clock::now();
        double oracleMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        bool ok = (R.total == B.total) && (R.pairCount.size() == B.pairCount.size());
        if (ok)
            for (size_t i = 0; i < R.pairCount.size(); ++i)
                if (R.pairCount[i] != B.pairCount[i]) { ok = false; break; }

        std::cout << "-------------------------------------------------------\n";
        if (ok)
            std::cout << "  [VERIFY] OK   : express == brute force  (total = " << R.total << ")\n";
        else
            std::cout << "  [VERIFY] FAIL : express total = " << R.total
                      << " , brute total = " << B.total << "\n";
        std::cout << "  [VERIFY] brute-force time: " << std::fixed << std::setprecision(2)
                  << oracleMs << " ms\n"
                  << "-------------------------------------------------------\n";
    }

    return R;
}