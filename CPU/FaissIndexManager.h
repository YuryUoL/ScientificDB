#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <unordered_set>
#include <numeric>
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
    std::vector<float> sdv;
    std::vector<float> srv;
    std::vector<float> pdd; // N x N row-major matrix of sorted squared distances
    bool computed = false;
};

class FaissIndexManager
{
public:
    FaissIndexManager() = default;
    ~FaissIndexManager() { clear(); }

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

    template <typename GetAtomsFunc>
    void computeDescriptorsForAll(size_t numEntries, GetAtomsFunc getAtoms)
    {
        if (numEntries == 0)
            return;

        auto tStart = std::chrono::high_resolution_clock::now();
        m_descriptors.assign(numEntries, MolecularDescriptors{});

#pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < static_cast<int>(numEntries); ++i)
        {
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

            float cx = 0.0f, cy = 0.0f, cz = 0.0f;
            for (const auto &a : atoms)
            {
                cx += a.x;
                cy += a.y;
                cz += a.z;
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
                    float dx = atoms[b].x - atoms[a].x;
                    float dy = atoms[b].y - atoms[a].y;
                    float dz = atoms[b].z - atoms[a].z;
                    pairwiseDistances.push_back(std::sqrt(dx * dx + dy * dy + dz * dz));
                }
            }
            std::sort(pairwiseDistances.begin(), pairwiseDistances.end());
            desc.sdv = std::move(pairwiseDistances);

            std::vector<float> centerDistances;
            centerDistances.reserve(n);
            for (const auto &a : atoms)
            {
                float dx = a.x - cx;
                float dy = a.y - cy;
                float dz = a.z - cz;
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

        auto tEnd = std::chrono::high_resolution_clock::now();
        double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        std::cout << "[FaissManager] Computed descriptors for " << numEntries << " molecules in "
                  << std::fixed << std::setprecision(2) << elapsedMs << " ms.\n";
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
            float sumSq = 0.0f;
            size_t n = std::min(descA.srv.size(), descB.srv.size());
            for (size_t i = 0; i < n; ++i)
            {
                float diff = descA.srv[i] - descB.srv[i];
                sumSq += diff * diff;
            }
            return sumSq;
        }
        else if (metricType == "sdv")
        {
            float sumSq = 0.0f;
            size_t n = std::min(descA.sdv.size(), descB.sdv.size());
            for (size_t i = 0; i < n; ++i)
            {
                float diff = descA.sdv[i] - descB.sdv[i];
                sumSq += diff * diff;
            }
            return sumSq;
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

            int totalGroupSize = groupGlobalIndices.size();
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
        if (m_descriptors.empty())
            return;
        m_sdvIndices.clear();
        m_srvIndices.clear();

        std::map<size_t, std::vector<size_t>> atomGroups;
        for (size_t i = 0; i < m_descriptors.size(); ++i)
        {
            if (m_descriptors[i].computed)
                atomGroups[m_descriptors[i].numAtoms].push_back(i);
        }

        for (const auto &[nAtoms, indices] : atomGroups)
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
        }
    }

    void clear()
    {
        m_descriptors.clear();
        m_sdvIndices.clear();
        m_srvIndices.clear();
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
        std::cout << "  SDV Dimensions  : " << desc.sdv.size() << " pairwise distances\n";
        std::cout << "  SRV Dimensions  : " << desc.srv.size() << " center distances\n";
        std::cout << "  PDD Dimensions  : " << n << " x " << n << " matrix (" << desc.pdd.size() << " elements)\n";
        std::cout << "-------------------------------------------------------\n";

        // Preview SRV
        std::cout << "  SRV (Sorted Center Distances): [";
        size_t srvLimit = std::min(desc.srv.size(), (size_t)5);
        for (size_t i = 0; i < srvLimit; ++i) {
            std::cout << std::fixed << std::setprecision(3) << desc.srv[i] << (i + 1 < srvLimit ? ", " : "");
        }
        if (desc.srv.size() > srvLimit) std::cout << ", ...";
        std::cout << "]\n";

        // Preview SDV
        std::cout << "  SDV (Sorted Pairwise Distances): [";
        size_t sdvLimit = std::min(desc.sdv.size(), (size_t)5);
        for (size_t i = 0; i < sdvLimit; ++i) {
            std::cout << std::fixed << std::setprecision(3) << desc.sdv[i] << (i + 1 < sdvLimit ? ", " : "");
        }
        if (desc.sdv.size() > sdvLimit) std::cout << ", ...";
        std::cout << "]\n";

        // Preview PDD Matrix (first 4x4 subset of actual distances via sqrt)
        std::cout << "  PDD Matrix Preview (" << std::min(n, (size_t)4) << "x" << std::min(n, (size_t)4) << "):\n";
        size_t rowLimit = std::min(n, (size_t)4);
        size_t colLimit = std::min(n, (size_t)4);
        for (size_t r = 0; r < rowLimit; ++r) {
            std::cout << "    Row " << std::setw(2) << r << ": [";
            for (size_t c = 0; c < colLimit; ++c) {
                std::cout << std::setw(8) << std::fixed << std::setprecision(2) << std::sqrt(desc.pdd[r * n + c]);
                if (c + 1 < colLimit) std::cout << ", ";
            }
            if (n > colLimit) std::cout << ", ...";
            std::cout << "]\n";
        }
        if (n > rowLimit) {
            std::cout << "    ...\n";
        }
        std::cout << "=======================================================\n\n";
    }

private:
    std::vector<MolecularDescriptors> m_descriptors;
    std::map<size_t, std::unique_ptr<faiss::IndexFlatL2>> m_sdvIndices;
    std::map<size_t, std::unique_ptr<faiss::IndexFlatL2>> m_srvIndices;
    mutable std::vector<KNNResult> m_lastSearchResults;

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