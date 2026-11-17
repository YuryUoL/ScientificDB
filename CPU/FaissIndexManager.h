#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <map>
#include <unordered_map>
#include <memory>
#include <unordered_set>
#include <numeric>
#include <cstdint>
#include <limits>
#include <omp.h>
#include <faiss/IndexFlat.h>
#include "PolyscopeBridge.h"
#include <fstream>
#include <atomic>
#include <chrono>
#include <execution>
#include "BottleneckEvaluator.h"

struct KNNResult
{
    size_t targetIndex;
    size_t matchedIndex;
    float distance;
};

struct MolecularDescriptors
{
    size_t numAtoms = 0;
    uint64_t compHash = 0;
    std::vector<float> sdv;
    std::vector<float> srv;
    std::vector<float> pdd; // N x N row-major matrix of sorted squared distances
    bool computed = false;
};

struct SrvSdvCandidate
{
    size_t idxA = 0;
    size_t idxB = 0;
    float srvLinf = 0.0f;
    float sdvLinf = 0.0f;
};

struct SortedBucket
{
    size_t numAtoms = 0;
    uint64_t compHash = 0;
    size_t sdvDim = 0;
    std::vector<size_t> globalIds; // Sorted by srv[0] ascending
    std::vector<float> srv0;       // Minimum SRV value per entry (for binary search / sliding window)
    std::vector<float> srvFlat;    // Contiguous [M * numAtoms]
    std::vector<float> sdvFlat;    // Contiguous [M * sdvDim]
};

class FaissIndexManager
{
public:
    FaissIndexManager() = default;
    ~FaissIndexManager() { clear(); }

    static uint64_t hashSymbolMultiset(const std::vector<std::string> &sortedSymbols)
    {
        uint64_t h = 14695981039346656037ULL; // FNV-1a 64-bit offset basis
        for (const auto &sym : sortedSymbols)
        {
            for (unsigned char c : sym)
            {
                h ^= static_cast<uint64_t>(c);
                h *= 1099511628211ULL;
            }
            h ^= static_cast<uint64_t>(',');
            h *= 1099511628211ULL;
        }
        return h;
    }

    template <typename AtomVec>
    static uint64_t computeCompositionHash(const AtomVec &atoms)
    {
        std::vector<std::string> syms;
        syms.reserve(atoms.size());
        for (const auto &a : atoms)
        {
            syms.emplace_back(a.symbol);
        }
        std::sort(syms.begin(), syms.end());
        return hashSymbolMultiset(syms);
    }

    size_t descriptorCount() const { return m_descriptors.size(); }

    size_t groupSize(size_t n) const
    {
        auto it = m_atomOnlyBuckets.find(n);
        if (it != m_atomOnlyBuckets.end())
            return it->second.globalIds.size();

        size_t count = 0;
        for (const auto &d : m_descriptors)
        {
            if (d.computed && d.numAtoms == n)
                count++;
        }
        return count;
    }

    const MolecularDescriptors *descriptor(size_t index) const
    {
        if (index >= m_descriptors.size() || !m_descriptors[index].computed)
            return nullptr;
        return &m_descriptors[index];
    }

    int parseK(const std::string &kStr, size_t totalSize) const
    {
        if (kStr.empty())
            return 5;
        if (kStr.back() == '%')
        {
            try
            {
                float pct = std::stof(kStr.substr(0, kStr.length() - 1));
                int k = static_cast<int>((pct / 100.0f) * totalSize);
                return std::max(1, std::min(k, static_cast<int>(totalSize)));
            }
            catch (...)
            {
                return 5;
            }
        }
        else
        {
            try
            {
                int k = std::stoi(kStr);
                return std::max(1, std::min(k, static_cast<int>(totalSize)));
            }
            catch (...)
            {
                return 5;
            }
        }
    }

    // Legacy single-stage SRV range scan (kept for compatibility)
    void rangeSearchSrvLinf(const std::vector<float> &querySrv, size_t n, float radius, size_t queryIdx, int numThreads, std::vector<size_t> &outIds) const
    {
        outIds.clear();
        auto it = m_atomOnlyBuckets.find(n);
        if (it == m_atomOnlyBuckets.end() || querySrv.empty())
            return;

        const auto &b = it->second;
        const size_t m = b.globalIds.size();
        if (m == 0)
            return;

        const float q0 = querySrv[0];
        auto loIt = std::lower_bound(b.srv0.begin(), b.srv0.end(), q0 - radius);
        auto hiIt = std::upper_bound(b.srv0.begin(), b.srv0.end(), q0 + radius);
        size_t lo = static_cast<size_t>(loIt - b.srv0.begin());
        size_t hi = static_cast<size_t>(hiIt - b.srv0.begin());
        if (lo >= hi)
            return;

        size_t span = hi - lo;
        outIds.resize(span);
        std::vector<char> keep(span, 0);

        int nt = numThreads > 0 ? numThreads : omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(nt) if (span > 256 && nt > 1)
        for (long long off = 0; off < static_cast<long long>(span); ++off)
        {
            size_t pos = lo + static_cast<size_t>(off);
            size_t globalId = b.globalIds[pos];
            if (globalId == queryIdx)
                continue;

            const float *candSrv = b.srvFlat.data() + pos * n;
            if (linfWithin(querySrv.data(), candSrv, n, radius) <= radius)
            {
                keep[off] = 1;
            }
        }

        size_t w = 0;
        for (size_t off = 0; off < span; ++off)
        {
            if (keep[off])
                outIds[w++] = b.globalIds[lo + off];
        }
        outIds.resize(w);
    }

    // Fused Stage 1 (SRV LInf) + Stage 2 (Composition + SDV LInf) for single query via sorted bucket binary search
    void rangeSearchSrvSdvSingle(size_t queryIdx, float rSrv, float rSdv, bool requireSameComp,
                                 int numThreads, bool parallel,
                                 size_t &outAfterSrv, std::vector<SrvSdvCandidate> &outCands) const
    {
        outAfterSrv = 0;
        outCands.clear();

        if (queryIdx >= m_descriptors.size() || !m_descriptors[queryIdx].computed)
            return;

        const auto &q = m_descriptors[queryIdx];
        const size_t n = q.numAtoms;
        if (n < 2 || q.srv.empty())
            return;

        const SortedBucket *bucket = findBucket(n, q.compHash, requireSameComp);
        if (!bucket || bucket->globalIds.empty())
            return;

        const float q0 = q.srv[0];
        auto loIt = std::lower_bound(bucket->srv0.begin(), bucket->srv0.end(), q0 - rSrv);
        auto hiIt = std::upper_bound(bucket->srv0.begin(), bucket->srv0.end(), q0 + rSrv);
        const size_t lo = static_cast<size_t>(loIt - bucket->srv0.begin());
        const size_t hi = static_cast<size_t>(hiIt - bucket->srv0.begin());
        if (lo >= hi)
            return;

        const size_t span = hi - lo;
        const size_t sdvDim = bucket->sdvDim;
        const float *qSrv = q.srv.data();
        const float *qSdv = q.sdv.data();

        std::vector<SrvSdvCandidate> slots(span);
        std::vector<char> passSrv(span, 0);
        std::vector<char> passSdv(span, 0);

        int nt = numThreads > 0 ? numThreads : omp_get_max_threads();
#pragma omp parallel for schedule(static) num_threads(nt) if (parallel && span > 256 && nt > 1)
        for (long long off = 0; off < static_cast<long long>(span); ++off)
        {
            const size_t pos = lo + static_cast<size_t>(off);
            const size_t candId = bucket->globalIds[pos];
            if (candId == queryIdx)
                continue;

            const float *cSrv = bucket->srvFlat.data() + pos * n;
            const float dSrv = linfWithin(qSrv, cSrv, n, rSrv);
            if (dSrv > rSrv)
                continue;

            passSrv[off] = 1;

            const float *cSdv = bucket->sdvFlat.data() + pos * sdvDim;
            const float dSdv = linfWithin(qSdv, cSdv, sdvDim, rSdv);
            if (dSdv > rSdv)
                continue;

            slots[off] = {queryIdx, candId, dSrv, dSdv};
            passSdv[off] = 1;
        }

        size_t srvCount = 0;
        for (size_t off = 0; off < span; ++off)
        {
            srvCount += static_cast<size_t>(passSrv[off]);
            if (passSdv[off])
                outCands.push_back(slots[off]);
        }
        outAfterSrv = srvCount;
    }

    // Fused Stage 1 + Stage 2 for ALL pairs (upper-triangular j > i within each sorted bucket)
    void rangeSearchSrvSdvAllPairs(float rSrv, float rSdv, bool requireSameComp,
                                   int numThreads, bool parallel,
                                   size_t &outTotalSrvPairs, std::vector<SrvSdvCandidate> &outCands) const
    {
        outTotalSrvPairs = 0;
        outCands.clear();

        std::vector<const SortedBucket *> buckets;
        if (requireSameComp)
        {
            buckets.reserve(m_compBuckets.size());
            for (const auto &kv : m_compBuckets)
            {
                if (kv.second.numAtoms >= 2 && kv.second.globalIds.size() >= 2)
                    buckets.push_back(&kv.second);
            }
        }
        else
        {
            buckets.reserve(m_atomOnlyBuckets.size());
            for (const auto &kv : m_atomOnlyBuckets)
            {
                if (kv.second.numAtoms >= 2 && kv.second.globalIds.size() >= 2)
                    buckets.push_back(&kv.second);
            }
        }

        if (buckets.empty())
            return;

        // Flatten bucket work items (bucketIdx, row i) for balanced dynamic scheduling across large buckets
        struct WorkItem
        {
            size_t bucketIdx;
            size_t i;
        };
        std::vector<WorkItem> work;
        for (size_t bIdx = 0; bIdx < buckets.size(); ++bIdx)
        {
            size_t m = buckets[bIdx]->globalIds.size();
            for (size_t i = 0; i + 1 < m; ++i)
            {
                work.push_back({bIdx, i});
            }
        }

        if (work.empty())
            return;

        int nt = parallel ? (numThreads > 0 ? numThreads : omp_get_max_threads()) : 1;
        std::vector<std::vector<SrvSdvCandidate>> threadCands(nt);
        size_t totalSrvSurv = 0;

#pragma omp parallel num_threads(nt) reduction(+ : totalSrvSurv)
        {
            int tid = omp_get_thread_num();
            auto &localOut = threadCands[tid];

#pragma omp for schedule(dynamic, 32)
            for (long long w = 0; w < static_cast<long long>(work.size()); ++w)
            {
                const SortedBucket &b = *buckets[work[w].bucketIdx];
                const size_t i = work[w].i;
                const size_t m = b.globalIds.size();
                const size_t n = b.numAtoms;
                const size_t sdvDim = b.sdvDim;

                const size_t idA = b.globalIds[i];
                const float srv0A = b.srv0[i];
                const float maxSrv0 = srv0A + rSrv;
                const float *srvA = b.srvFlat.data() + i * n;
                const float *sdvA = b.sdvFlat.data() + i * sdvDim;

                for (size_t j = i + 1; j < m; ++j)
                {
                    if (b.srv0[j] > maxSrv0)
                        break; // Sorted by srv0 ascending: no further j can match

                    const float *srvB = b.srvFlat.data() + j * n;
                    const float dSrv = linfWithin(srvA, srvB, n, rSrv);
                    if (dSrv > rSrv)
                        continue;

                    totalSrvSurv += 1;

                    const float *sdvB = b.sdvFlat.data() + j * sdvDim;
                    const float dSdv = linfWithin(sdvA, sdvB, sdvDim, rSdv);
                    if (dSdv > rSdv)
                        continue;

                    size_t idB = b.globalIds[j];
                    if (idA < idB)
                        localOut.push_back({idA, idB, dSrv, dSdv});
                    else
                        localOut.push_back({idB, idA, dSrv, dSdv});
                }
            }
        }

        outTotalSrvPairs = totalSrvSurv;
        size_t totalSurvivors = 0;
        for (const auto &v : threadCands)
            totalSurvivors += v.size();
        outCands.reserve(totalSurvivors);
        for (auto &v : threadCands)
        {
            outCands.insert(outCands.end(), v.begin(), v.end());
        }
    }

    bool searchKNN(size_t queryIdx, const std::string &kStr, const std::string &metricType,
                   const std::string &csvOutputPath = "", bool restrictToLast = false) const
    {
        if (queryIdx >= m_descriptors.size() || !m_descriptors[queryIdx].computed)
        {
            std::cout << "[FaissManager] Error: Query index " << queryIdx
                      << " out of bounds or descriptors not computed.\n";
            return false;
        }

        const auto &targetDesc = m_descriptors[queryIdx];
        size_t nAtoms = targetDesc.numAtoms;

        std::vector<size_t> groupGlobalIndices;
        groupGlobalIndices.reserve(m_descriptors.size());
        for (size_t i = 0; i < m_descriptors.size(); ++i)
        {
            if (m_descriptors[i].computed && m_descriptors[i].numAtoms == nAtoms)
            {
                groupGlobalIndices.push_back(i);
            }
        }

        size_t totalGroupSize = groupGlobalIndices.size();
        if (totalGroupSize == 0)
        {
            std::cout << "[FaissManager] Error: No molecules in atom group " << nAtoms << ".\n";
            return false;
        }

        int parsedK = parseK(kStr, totalGroupSize);

        const faiss::IndexFlatL2 *targetIndex = nullptr;
        const std::vector<float> *queryVec = nullptr;

        if (metricType == "sdv")
        {
            auto it = m_sdvIndices.find(nAtoms);
            if (it != m_sdvIndices.end())
            {
                targetIndex = it->second.get();
                queryVec = &targetDesc.sdv;
            }
        }
        else if (metricType == "srv")
        {
            auto it = m_srvIndices.find(nAtoms);
            if (it != m_srvIndices.end())
            {
                targetIndex = it->second.get();
                queryVec = &targetDesc.srv;
            }
        }

        if (!targetIndex || !queryVec || queryVec->empty())
        {
            std::cout << "[FaissManager] Error: No valid index found for atom count group (" << nAtoms << " atoms).\n";
            return false;
        }

        double tStart = omp_get_wtime();
        std::vector<KNNResult> results;

        if (!restrictToLast || m_lastSearchResults.empty())
        {
            int fetchK = std::min(parsedK + 1, static_cast<int>(totalGroupSize));
            std::vector<float> distances(fetchK);
            std::vector<faiss::idx_t> indices(fetchK);

            targetIndex->search(1, queryVec->data(), fetchK, distances.data(), indices.data());

            results.reserve(parsedK);
            for (int r = 0; r < fetchK; ++r)
            {
                faiss::idx_t localIdx = indices[r];
                size_t globalIdx = (localIdx >= 0 && localIdx < static_cast<faiss::idx_t>(groupGlobalIndices.size()))
                                       ? groupGlobalIndices[localIdx]
                                       : 0;

                if (globalIdx == queryIdx)
                    continue;

                results.push_back({queryIdx, globalIdx, distances[r]});
                if (results.size() == static_cast<size_t>(parsedK))
                    break;
            }
        }
        else
        {
            int searchK = static_cast<int>(totalGroupSize);
            std::vector<float> distances(searchK);
            std::vector<faiss::idx_t> indices(searchK);

            targetIndex->search(1, queryVec->data(), searchK, distances.data(), indices.data());

            std::unordered_set<size_t> allowedSet;
            for (const auto &res : m_lastSearchResults)
                allowedSet.insert(res.matchedIndex);

            std::vector<KNNResult> filteredResults;
            for (int r = 0; r < searchK; ++r)
            {
                faiss::idx_t localIdx = indices[r];
                if (localIdx < 0 || localIdx >= static_cast<faiss::idx_t>(groupGlobalIndices.size()))
                    continue;
                size_t globalIdx = groupGlobalIndices[localIdx];

                if (globalIdx == queryIdx)
                    continue;

                if (allowedSet.find(globalIdx) != allowedSet.end())
                {
                    filteredResults.push_back({queryIdx, globalIdx, distances[r]});
                }
            }

            int actualK = std::min(parsedK, static_cast<int>(filteredResults.size()));
            results.assign(filteredResults.begin(), filteredResults.begin() + actualK);
        }

        double searchTimeSec = omp_get_wtime() - tStart;
        m_lastSearchResults = results;

        std::cout << "\n[FaissManager] Search completed in " << std::fixed << std::setprecision(3)
                  << (searchTimeSec * 1000.0) << " ms for query molecule " << queryIdx
                  << " (Retrieved " << results.size() << " neighbors using " << metricType << ").\n";

        std::cout << "-----------------------------------------------------------------------\n";
        std::cout << std::left << std::setw(8) << "Rank"
                  << std::setw(14) << "Global Index"
                  << std::setw(14) << "num_atom_A"
                  << std::setw(14) << "num_atom_B"
                  << std::setw(20) << "L2 Distance" << "\n";
        std::cout << "-----------------------------------------------------------------------\n";

        int previewLimit = std::min(static_cast<int>(results.size()), 5);
        for (int r = 0; r < previewLimit; ++r)
        {
            size_t natA = m_descriptors[results[r].targetIndex].numAtoms;
            size_t natB = m_descriptors[results[r].matchedIndex].numAtoms;
            std::cout << std::left << std::setw(8) << (r + 1)
                      << std::setw(14) << results[r].matchedIndex
                      << std::setw(14) << natA
                      << std::setw(14) << natB
                      << std::setw(20) << std::fixed << std::setprecision(6) << results[r].distance << "\n";
        }
        if (results.size() > static_cast<size_t>(previewLimit))
        {
            std::cout << "  ... (showing top " << previewLimit << " of " << results.size() << " neighbors; use 'dist_summary' for full histogram/stats)\n";
        }
        std::cout << "-----------------------------------------------------------------------\n\n";

        if (!csvOutputPath.empty())
        {
            std::ofstream csv(csvOutputPath);
            if (csv.is_open())
            {
                csv << "rank,query_index,matched_index,num_atom_A,num_atom_B,metric_type,l2_distance\n";
                for (size_t r = 0; r < results.size(); ++r)
                {
                    size_t natA = m_descriptors[results[r].targetIndex].numAtoms;
                    size_t natB = m_descriptors[results[r].matchedIndex].numAtoms;
                    csv << (r + 1) << "," << results[r].targetIndex << ","
                        << results[r].matchedIndex << "," << natA << "," << natB << ","
                        << metricType << ","
                        << std::fixed << std::setprecision(6) << results[r].distance << "\n";
                }
                std::cout << "[FaissManager] Exported " << results.size() << " results to CSV: " << csvOutputPath << "\n";
            }
        }

        return true;
    }

    void printDistanceDistribution() const
    {
        if (m_lastSearchResults.empty())
        {
            std::cout << "[FaissManager] No recent search results to analyze. Run a 'knn', 'chain', or 'knn_all' search first.\n";
            return;
        }

        std::vector<float> dists;
        dists.reserve(m_lastSearchResults.size());
        for (const auto &r : m_lastSearchResults)
            dists.push_back(r.distance);

        float minD = *std::min_element(dists.begin(), dists.end());
        float maxD = *std::max_element(dists.begin(), dists.end());
        float sum = std::accumulate(dists.begin(), dists.end(), 0.0f);
        float mean = sum / dists.size();

        float sqSum = 0.0f;
        for (float d : dists)
            sqSum += (d - mean) * (d - mean);
        float stdDev = std::sqrt(sqSum / dists.size());

        std::cout << "\n=======================================================\n";
        std::cout << "      DISTANCE DISTRIBUTION ANALYSIS (LOG-SCALE)       \n";
        std::cout << "=======================================================\n";
        std::cout << "  Sample Count : " << dists.size() << "\n";
        std::cout << "  Min Distance : " << std::fixed << std::setprecision(6) << minD << "\n";
        std::cout << "  Max Distance : " << maxD << "\n";
        std::cout << "  Mean (mu)    : " << mean << "\n";
        std::cout << "  StdDev (sig) : " << stdDev << "\n";
        std::cout << "-------------------------------------------------------\n";
        std::cout << "  Log-Scale ASCII Distance Histogram (10 Bins):\n";

        int numBins = 10;
        std::vector<int> bins(numBins, 0);

        float eps = 1e-7f;
        float logMin = std::log(std::max(eps, minD));
        float logMax = std::log(std::max(eps, maxD));
        if (logMax <= logMin)
            logMax = logMin + 1.0f;
        float logWidth = (logMax - logMin) / numBins;

        for (float d : dists)
        {
            float ld = std::log(std::max(eps, d));
            int b = static_cast<int>((ld - logMin) / logWidth);
            if (b >= numBins)
                b = numBins - 1;
            if (b < 0)
                b = 0;
            bins[b]++;
        }

        int maxBinCount = *std::max_element(bins.begin(), bins.end());
        int maxBarWidth = 40;

        for (int i = 0; i < numBins; ++i)
        {
            float binLow = std::exp(logMin + i * logWidth);
            float binHigh = std::exp(logMin + (i + 1) * logWidth);
            int barLen = (maxBinCount > 0) ? static_cast<int>((static_cast<float>(bins[i]) / maxBinCount) * maxBarWidth) : 0;

            std::cout << "  [" << std::setw(10) << std::scientific << std::setprecision(2) << binLow
                      << " - " << std::setw(10) << binHigh << "] | ";
            for (int s = 0; s < barLen; ++s)
                std::cout << "=";
            std::cout << " (" << bins[i] << ")\n";
        }
        std::cout << "=======================================================\n\n";
    }

    // Incremental range computation: computes descriptors for [startIdx, endIdx) and rebuilds sorted buckets + FAISS indices
    template <typename GetAtomsFunc>
    void computeDescriptorsRange(size_t startIdx, size_t endIdx, GetAtomsFunc getAtoms, bool verbose = true)
    {
        if (endIdx == 0)
        {
            clear();
            return;
        }

        if (m_descriptors.size() != endIdx)
        {
            m_descriptors.resize(endIdx);
        }

        if (startIdx < endIdx)
        {
            auto tStart = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 16)
            for (int i = static_cast<int>(startIdx); i < static_cast<int>(endIdx); ++i)
            {
                if (m_descriptors[i].computed)
                    continue;

                const auto &atoms = getAtoms(i);
                MolecularDescriptors desc;
                size_t n = atoms.size();
                desc.numAtoms = n;
                if (n == 0)
                {
                    desc.computed = true;
                    m_descriptors[i] = std::move(desc);
                    continue;
                }

                desc.compHash = computeCompositionHash(atoms);

                float cx = 0.0f, cy = 0.0f, cz = 0.0f;
                for (const auto &a : atoms)
                {
                    cx += static_cast<float>(a.x);
                    cy += static_cast<float>(a.y);
                    cz += static_cast<float>(a.z);
                }
                cx /= n;
                cy /= n;
                cz /= n;

                size_t numPairs = (n > 1) ? (n * (n - 1)) / 2 : 0;
                std::vector<float> pairwiseDistances;
                pairwiseDistances.reserve(numPairs);
                for (size_t a = 0; a < n; ++a)
                {
                    for (size_t b = a + 1; b < n; ++b)
                    {
                        float dx = static_cast<float>(atoms[b].x - atoms[a].x);
                        float dy = static_cast<float>(atoms[b].y - atoms[a].y);
                        float dz = static_cast<float>(atoms[b].z - atoms[a].z);
                        pairwiseDistances.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
                    }
                }
                std::sort(pairwiseDistances.begin(), pairwiseDistances.end());
                desc.sdv = std::move(pairwiseDistances);

                std::vector<float> centerDistances;
                centerDistances.reserve(n);
                for (const auto &a : atoms)
                {
                    float dx = static_cast<float>(a.x) - cx;
                    float dy = static_cast<float>(a.y) - cy;
                    float dz = static_cast<float>(a.z) - cz;
                    centerDistances.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
                }
                std::sort(centerDistances.begin(), centerDistances.end());
                desc.srv = std::move(centerDistances);

                // Compute PDD Descriptor Matrix
                BottleneckEvaluator::computePDD(atoms, desc.pdd);

                desc.computed = true;
                m_descriptors[i] = std::move(desc);
            }

            buildIndices();

            if (verbose)
            {
                auto tEnd = std::chrono::high_resolution_clock::now();
                double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
                std::cout << "[FaissManager] Computed descriptors for " << (endIdx - startIdx)
                          << " molecules (total indexed: " << endIdx << ") in "
                          << std::fixed << std::setprecision(2) << elapsedMs << " ms.\n";
            }
        }
        else
        {
            buildIndices();
        }
    }

    template <typename GetAtomsFunc>
    void computeDescriptorsForAll(size_t numEntries, GetAtomsFunc getAtoms)
    {
        if (numEntries == 0)
            return;
        m_descriptors.assign(numEntries, MolecularDescriptors{});
        computeDescriptorsRange(0, numEntries, getAtoms, true);
    }

    // Keeps existing computed descriptors intact when appending or removing entries
    template <typename GetAtomsFunc>
    void syncWithEntries(size_t totalEntries, GetAtomsFunc getAtoms)
    {
        if (totalEntries == 0)
        {
            clear();
            return;
        }
        size_t firstUncomputed = totalEntries;
        for (size_t i = 0; i < totalEntries; ++i)
        {
            if (i >= m_descriptors.size() || !m_descriptors[i].computed)
            {
                firstUncomputed = i;
                break;
            }
        }
        computeDescriptorsRange(firstUncomputed, totalEntries, getAtoms, firstUncomputed < totalEntries);
    }

    void removeEntryAt(size_t index)
    {
        if (index < m_descriptors.size())
        {
            m_descriptors.erase(m_descriptors.begin() + index);
            buildIndices();
        }
    }

   float computePairDistance(size_t idxA, size_t idxB, const std::string &metricType) const
    {
        if (idxA >= m_descriptors.size() || idxB >= m_descriptors.size() ||
            !m_descriptors[idxA].computed || !m_descriptors[idxB].computed)
        {
            std::cout << "[FaissManager] Error: Descriptors not computed for target indices.\n";
            return -1.0f;
        }

        const auto &descA = m_descriptors[idxA];
        const auto &descB = m_descriptors[idxB];

        if (descA.numAtoms != descB.numAtoms)
        {
            std::cout << "[FaissManager] Warning: Atom counts differ ("
                      << descA.numAtoms << " vs " << descB.numAtoms << ").\n";
        }

        if (metricType == "srv")
        {
            float maxDiff = 0.0f;
            size_t n = std::min(descA.srv.size(), descB.srv.size());
            for (size_t i = 0; i < n; ++i)
            {
                float diff = std::fabs(descA.srv[i] - descB.srv[i]);
                if (diff > maxDiff) maxDiff = diff;
            }
            return maxDiff;
        }
        else if (metricType == "sdv")
        {
            float maxDiff = 0.0f;
            size_t n = std::min(descA.sdv.size(), descB.sdv.size());
            for (size_t i = 0; i < n; ++i)
            {
                float diff = std::fabs(descA.sdv[i] - descB.sdv[i]);
                if (diff > maxDiff) maxDiff = diff;
            }
            return maxDiff;
        }
        else if (metricType == "pdd" || metricType == "bottleneck")
        {
            size_t n = descA.numAtoms;
            if (n > BottleneckEvaluator::MAX_VERTICES)
            {
                std::cout << "[FaissManager] Error: Atom count exceeds MAX_VERTICES (64).\n";
                return -1.0f;
            }
            float costMatrix[BottleneckEvaluator::MAX_VERTICES * BottleneckEvaluator::MAX_VERTICES];
            BottleneckEvaluator::computeCostMatrix(descA.pdd.data(), descB.pdd.data(), n, costMatrix);
            return BottleneckEvaluator::computeFromCostMatrix(costMatrix, n);
        }

        std::cout << "[FaissManager] Unknown metric type: " << metricType << "\n";
        return -1.0f;
    }
    bool searchKNNForAll(const std::string &kStr, const std::string &metricType, const std::string &csvOutputPath) const
    {
        if (m_descriptors.empty())
        {
            std::cout << "[FaissManager] Error: No descriptors computed.\n";
            return false;
        }

        double tStart = omp_get_wtime();
        size_t totalQueries = m_descriptors.size();
        size_t successCount = 0;

        std::cout << "\n[FaissManager] Running global batch " << metricType << " KNN search for all " << totalQueries << " molecules...\n";

        m_lastSearchResults.clear();
        std::atomic<size_t> processedCount{0};

        std::vector<std::vector<KNNResult>> allThreadsResults(totalQueries);

#pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < static_cast<int>(totalQueries); ++i)
        {
            if (!m_descriptors[i].computed)
                continue;

            size_t nAtoms = m_descriptors[i].numAtoms;
            const faiss::IndexFlatL2 *targetIndex = nullptr;
            const std::vector<float> *queryVec = nullptr;

            auto sdvIt = m_sdvIndices.find(nAtoms);
            auto srvIt = m_srvIndices.find(nAtoms);

            if (metricType == "sdv" && sdvIt != m_sdvIndices.end())
            {
                targetIndex = sdvIt->second.get();
                queryVec = &m_descriptors[i].sdv;
            }
            else if (metricType == "srv" && srvIt != m_srvIndices.end())
            {
                targetIndex = srvIt->second.get();
                queryVec = &m_descriptors[i].srv;
            }

            if (!targetIndex || !queryVec || queryVec->empty())
                continue;

            std::vector<size_t> groupGlobalIndices;
            for (size_t j = 0; j < totalQueries; ++j)
            {
                if (m_descriptors[j].computed && m_descriptors[j].numAtoms == nAtoms)
                {
                    groupGlobalIndices.push_back(j);
                }
            }

            int totalGroupSize = static_cast<int>(groupGlobalIndices.size());
            int parsedK = parseK(kStr, totalGroupSize);
            int fetchK = std::min(std::max(parsedK * 2, 100), totalGroupSize);

            std::vector<float> distances(fetchK);
            std::vector<faiss::idx_t> indices(fetchK);

            targetIndex->search(1, queryVec->data(), fetchK, distances.data(), indices.data());

            std::vector<KNNResult> localRes;
            for (int r = 0; r < fetchK; ++r)
            {
                faiss::idx_t localIdx = indices[r];
                if (localIdx < 0 || localIdx >= static_cast<faiss::idx_t>(groupGlobalIndices.size()))
                    continue;
                size_t globalIdx = groupGlobalIndices[localIdx];

                if (globalIdx == static_cast<size_t>(i))
                    continue;

                localRes.push_back({static_cast<size_t>(i), globalIdx, distances[r]});
            }

            allThreadsResults[i] = std::move(localRes);

#pragma omp critical
            {
                successCount++;
                size_t currentDone = ++processedCount;
                if (currentDone % 50 == 0 || currentDone == totalQueries)
                {
                    printProgressBar(currentDone, totalQueries);
                }
            }
        }

        std::cout << "\n[FaissManager] Aggregating search results across threads...\n";
        size_t totalEstimatedPairs = 0;
        for (const auto &resVec : allThreadsResults)
        {
            totalEstimatedPairs += resVec.size();
        }
        m_lastSearchResults.reserve(totalEstimatedPairs);
        for (const auto &resVec : allThreadsResults)
        {
            for (const auto &res : resVec)
            {
                m_lastSearchResults.push_back(res);
            }
        }

        std::cout << "[FaissManager] Running parallel sort on " << m_lastSearchResults.size() << " candidate pairs...\n";
        std::sort(std::execution::par, m_lastSearchResults.begin(), m_lastSearchResults.end(),
                  [](const KNNResult &a, const KNNResult &b)
                  {
                      return a.distance < b.distance;
                  });

        size_t totalCollected = m_lastSearchResults.size();
        size_t globalLimit = totalCollected;

        if (!kStr.empty() && kStr.back() == '%')
        {
            try
            {
                float pct = std::stof(kStr.substr(0, kStr.length() - 1));
                globalLimit = static_cast<size_t>((pct / 100.0f) * totalCollected);
            }
            catch (...)
            {
            }
        }
        else
        {
            try
            {
                globalLimit = static_cast<size_t>(std::stoul(kStr));
            }
            catch (...)
            {
            }
        }

        globalLimit = std::max(size_t(1), std::min(globalLimit, totalCollected));
        m_lastSearchResults.resize(globalLimit);

        std::cout << "\n------------------------------------------------------------------------------------\n";
        std::cout << "  Global Top " << globalLimit << " Closest Pairs Across Entire Dataset\n";
        std::cout << "------------------------------------------------------------------------------------\n";
        std::cout << std::left << std::setw(12) << "GlobalRank"
                  << std::setw(14) << "Query Idx"
                  << std::setw(14) << "Matched Idx"
                  << std::setw(14) << "num_atom_A"
                  << std::setw(14) << "num_atom_B"
                  << std::setw(16) << "L2 Distance" << "\n";
        std::cout << "------------------------------------------------------------------------------------\n";

        size_t previewLimit = std::min(m_lastSearchResults.size(), (size_t)10);
        for (size_t r = 0; r < previewLimit; ++r)
        {
            size_t natA = m_descriptors[m_lastSearchResults[r].targetIndex].numAtoms;
            size_t natB = m_descriptors[m_lastSearchResults[r].matchedIndex].numAtoms;
            std::cout << std::left << std::setw(12) << (r + 1)
                      << std::setw(14) << m_lastSearchResults[r].targetIndex
                      << std::setw(14) << m_lastSearchResults[r].matchedIndex
                      << std::setw(14) << natA
                      << std::setw(14) << natB
                      << std::setw(16) << std::fixed << std::setprecision(6) << m_lastSearchResults[r].distance << "\n";
        }
        if (m_lastSearchResults.size() > previewLimit)
        {
            std::cout << "  ... (showing top " << previewLimit << " of " << m_lastSearchResults.size() << " total global pairs)\n";
        }
        std::cout << "------------------------------------------------------------------------------------\n";

        if (!csvOutputPath.empty())
        {
            std::cout << "[FaissManager] Writing CSV export to: " << csvOutputPath << "\n";
            std::ofstream csv(csvOutputPath);
            if (csv.is_open())
            {
                csv << "global_rank,query_index,matched_index,num_atom_A,num_atom_B,metric_type,l2_distance\n";
                for (size_t r = 0; r < m_lastSearchResults.size(); ++r)
                {
                    size_t natA = m_descriptors[m_lastSearchResults[r].targetIndex].numAtoms;
                    size_t natB = m_descriptors[m_lastSearchResults[r].matchedIndex].numAtoms;
                    csv << (r + 1) << "," << m_lastSearchResults[r].targetIndex << ","
                        << m_lastSearchResults[r].matchedIndex << "," << natA << "," << natB << ","
                        << metricType << ","
                        << std::fixed << std::setprecision(6) << m_lastSearchResults[r].distance << "\n";
                }
                std::cout << "[FaissManager] Exported exactly " << m_lastSearchResults.size() << " globally sorted rows successfully.\n";
            }
        }

        double elapsedSec = omp_get_wtime() - tStart;
        std::cout << "\n[FaissManager] Global batch KNN completed in "
                  << std::fixed << std::setprecision(3) << (elapsedSec * 1000.0) << " ms total.\n";

        return true;
    }

    void buildIndices()
    {
        m_sdvIndices.clear();
        m_srvIndices.clear();
        m_atomOnlyBuckets.clear();
        m_compBuckets.clear();

        if (m_descriptors.empty())
            return;

        std::map<size_t, std::vector<size_t>> atomGroups;
        std::unordered_map<BucketKey, std::vector<size_t>, BucketKeyHash> compGroups;

        for (size_t i = 0; i < m_descriptors.size(); ++i)
        {
            if (!m_descriptors[i].computed)
                continue;
            size_t n = m_descriptors[i].numAtoms;
            atomGroups[n].push_back(i);
            if (n >= 2)
            {
                compGroups[{n, m_descriptors[i].compHash}].push_back(i);
            }
        }

        // 1. Build FAISS L2 indices + sorted atom-only flat buckets
        for (auto &[nAtoms, indices] : atomGroups)
        {
            if (indices.empty() || nAtoms == 0)
                continue;

            size_t numPairs = (nAtoms > 1) ? (nAtoms * (nAtoms - 1)) / 2 : 0;
            if (numPairs > 0)
            {
                auto sdvIdx = std::make_unique<faiss::IndexFlatL2>(numPairs);
                std::vector<float> sdvFlat;
                sdvFlat.reserve(indices.size() * numPairs);
                for (size_t idx : indices)
                    sdvFlat.insert(sdvFlat.end(), m_descriptors[idx].sdv.begin(), m_descriptors[idx].sdv.end());
                sdvIdx->add(indices.size(), sdvFlat.data());
                m_sdvIndices[nAtoms] = std::move(sdvIdx);
            }

            auto srvIdx = std::make_unique<faiss::IndexFlatL2>(nAtoms);
            std::vector<float> srvFlat;
            srvFlat.reserve(indices.size() * nAtoms);
            for (size_t idx : indices)
                srvFlat.insert(srvFlat.end(), m_descriptors[idx].srv.begin(), m_descriptors[idx].srv.end());
            srvIdx->add(indices.size(), srvFlat.data());
            m_srvIndices[nAtoms] = std::move(srvIdx);

            if (nAtoms >= 2)
            {
                m_atomOnlyBuckets[nAtoms] = buildSortedBucket(nAtoms, 0, indices);
            }
        }

        // 2. Build sorted (numAtoms, compHash) flat buckets for duplicate filtering
        for (auto &[key, indices] : compGroups)
        {
            m_compBuckets[key] = buildSortedBucket(key.numAtoms, key.compHash, indices);
        }
    }

    void clear()
    {
        m_descriptors.clear();
        m_sdvIndices.clear();
        m_srvIndices.clear();
        m_atomOnlyBuckets.clear();
        m_compBuckets.clear();
        m_lastSearchResults.clear();
    }

    void printDescriptors(size_t index) const
    {
        if (index >= m_descriptors.size() || !m_descriptors[index].computed)
        {
            std::cout << "  FAISS Descriptors: [Not computed / Flushed]\n";
            return;
        }

        const auto &desc = m_descriptors[index];
        size_t n = desc.numAtoms;

        std::cout << "\n=======================================================\n";
        std::cout << "           MOLECULAR DESCRIPTORS [Index " << index << "]       \n";
        std::cout << "=======================================================\n";
        std::cout << "  Atom Count (N)  : " << n << "\n";
        std::cout << "  Comp Hash       : 0x" << std::hex << desc.compHash << std::dec << "\n";
        std::cout << "  SDV Dimensions  : " << desc.sdv.size() << " pairwise distances\n";
        std::cout << "  SRV Dimensions  : " << desc.srv.size() << " center distances\n";
        std::cout << "  PDD Dimensions  : " << n << " x " << n << " matrix (" << desc.pdd.size() << " elements)\n";
        std::cout << "-------------------------------------------------------\n";

        std::cout << "  SRV (Sorted Center Distances): [";
        size_t srvLimit = std::min(desc.srv.size(), (size_t)5);
        for (size_t i = 0; i < srvLimit; ++i)
        {
            std::cout << std::fixed << std::setprecision(3) << desc.srv[i] << (i + 1 < srvLimit ? ", " : "");
        }
        if (desc.srv.size() > srvLimit)
            std::cout << ", ...";
        std::cout << "]\n";

        std::cout << "  SDV (Sorted Pairwise Distances): [";
        size_t sdvLimit = std::min(desc.sdv.size(), (size_t)5);
        for (size_t i = 0; i < sdvLimit; ++i)
        {
            std::cout << std::fixed << std::setprecision(3) << desc.sdv[i] << (i + 1 < sdvLimit ? ", " : "");
        }
        if (desc.sdv.size() > sdvLimit)
            std::cout << ", ...";
        std::cout << "]\n";

        std::cout << "  PDD Matrix Preview (" << std::min(n, (size_t)4) << "x" << std::min(n, (size_t)4) << "):\n";
        size_t rowLimit = std::min(n, (size_t)4);
        size_t colLimit = std::min(n, (size_t)4);
        for (size_t r = 0; r < rowLimit; ++r)
        {
            std::cout << "    Row " << std::setw(2) << r << ": [";
            for (size_t c = 0; c < colLimit; ++c)
            {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << std::sqrt(desc.pdd[r * n + c]);
                if (c + 1 < colLimit)
                    std::cout << ", ";
            }
            if (n > colLimit)
                std::cout << ", ...";
            std::cout << "]\n";
        }
        if (n > rowLimit)
        {
            std::cout << "    ...\n";
        }
        std::cout << "=======================================================\n\n";
    }

private:
    struct BucketKey
    {
        size_t numAtoms;
        uint64_t compHash;
        bool operator==(const BucketKey &o) const
        {
            return numAtoms == o.numAtoms && compHash == o.compHash;
        }
    };

    struct BucketKeyHash
    {
        size_t operator()(const BucketKey &k) const
        {
            return std::hash<size_t>{}(k.numAtoms) ^ (k.compHash * 0x9e3779b97f4a7c15ULL);
        }
    };

    std::vector<MolecularDescriptors> m_descriptors;
    std::map<size_t, std::unique_ptr<faiss::IndexFlatL2>> m_sdvIndices;
    std::map<size_t, std::unique_ptr<faiss::IndexFlatL2>> m_srvIndices;
    std::unordered_map<size_t, SortedBucket> m_atomOnlyBuckets;
    std::unordered_map<BucketKey, SortedBucket, BucketKeyHash> m_compBuckets;
    mutable std::vector<KNNResult> m_lastSearchResults;

    static inline float linfWithin(const float *a, const float *b, size_t len, float limit)
    {
        float m = 0.0f;
        for (size_t k = 0; k < len; ++k)
        {
            float d = std::fabs(a[k] - b[k]);
            if (d > limit)
                return std::numeric_limits<float>::infinity();
            if (d > m)
                m = d;
        }
        return m;
    }

    const SortedBucket *findBucket(size_t numAtoms, uint64_t compHash, bool requireSameComp) const
    {
        if (requireSameComp)
        {
            auto it = m_compBuckets.find({numAtoms, compHash});
            return (it != m_compBuckets.end()) ? &it->second : nullptr;
        }
        auto it = m_atomOnlyBuckets.find(numAtoms);
        return (it != m_atomOnlyBuckets.end()) ? &it->second : nullptr;
    }

    SortedBucket buildSortedBucket(size_t nAtoms, uint64_t compHash, std::vector<size_t> &indices) const
    {
        std::sort(indices.begin(), indices.end(), [this](size_t a, size_t b)
                  {
                      float sa = m_descriptors[a].srv.empty() ? 0.0f : m_descriptors[a].srv[0];
                      float sb = m_descriptors[b].srv.empty() ? 0.0f : m_descriptors[b].srv[0];
                      if (sa != sb)
                          return sa < sb;
                      return a < b;
                  });

        SortedBucket b;
        b.numAtoms = nAtoms;
        b.compHash = compHash;
        b.sdvDim = (nAtoms > 1) ? (nAtoms * (nAtoms - 1)) / 2 : 0;
        const size_t m = indices.size();

        b.globalIds = indices;
        b.srv0.resize(m);
        b.srvFlat.resize(m * nAtoms);
        b.sdvFlat.resize(m * b.sdvDim);

        for (size_t row = 0; row < m; ++row)
        {
            const auto &desc = m_descriptors[indices[row]];
            b.srv0[row] = desc.srv.empty() ? 0.0f : desc.srv[0];
            std::copy(desc.srv.begin(), desc.srv.end(), b.srvFlat.begin() + row * nAtoms);
            std::copy(desc.sdv.begin(), desc.sdv.end(), b.sdvFlat.begin() + row * b.sdvDim);
        }
        return b;
    }

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