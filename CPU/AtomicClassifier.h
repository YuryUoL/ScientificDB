#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <atomic>
#include <sstream>
#include <omp.h>
#include "CPU/Visualization/PolyscopeBridge.h"
#include "Aligners/CloudAlignmentEvaluator.h"
#include "Aligners/AtomicCloudAlignmentEvaluator.h"

struct AtomicClassifyParams
{
    size_t N = 4;                           // Number of nearest neighbor atoms to compare (excluding center atom)
    float epsilon = 0.10f;                  // Distance threshold (Angstroms)
    bool considerSame = false;              // False: only compare clouds with different center AtomType; True: all pairs
    bool optimal = false;                   // False: fast early-exit aligner (<= eps); True: exact minimum bottleneck
    float srvFactor = 1.0f;                 // Stage 1 SRV L_inf threshold multiplier
    float alignFactor = 1.0f;               // Stage 2 Aligner bottleneck threshold multiplier
    float relTol = 1e-4f;                   // Float-noise slack on thresholds
    bool requireSameNeighborSpecies = false; // Require matched neighbor atoms to have identical element symbols
    int refineIterations = 0;               // Aligner Kabsch refinement rounds
    size_t maxMatches = 0;                  // Keep only best K matches (0 = all)
    int numThreads = 0;                     // 0 = OpenMP default
    bool parallel = true;                   // Enable/disable OpenMP parallelism

    bool set(const std::string &k, const std::string &v)
    {
        try
        {
            auto b = [&]()
            { return v == "1" || v == "on" || v == "true"; };
            if (k == "n" || k == "N")
                N = static_cast<size_t>(std::stoul(v));
            else if (k == "epsilon" || k == "eps")
                epsilon = std::stof(v);
            else if (k == "same" || k == "consider_same" || k == "considerSame")
                considerSame = b();
            else if (k == "optimal" || k == "opt" || k == "exact")
                optimal = b();
            else if (k == "srvf")
                srvFactor = std::stof(v);
            else if (k == "alignf")
                alignFactor = std::stof(v);
            else if (k == "tol")
                relTol = std::stof(v);
            else if (k == "species")
                requireSameNeighborSpecies = b();
            else if (k == "refine")
                refineIterations = std::stoi(v);
            else if (k == "max")
                maxMatches = static_cast<size_t>(std::stoul(v));
            else if (k == "threads")
                numThreads = std::stoi(v);
            else if (k == "parallel")
                parallel = b();
            else
                return false;
            return true;
        }
        catch (...)
        {
            return false;
        }
    }
};

struct AtomicClassifyMatch
{
    size_t atomicIdxA = 0;
    size_t atomicIdxB = 0;
    size_t datasetIdxA = 0;
    size_t atomIdxA = 0;
    std::string atomTypeA;
    size_t datasetIdxB = 0;
    size_t atomIdxB = 0;
    std::string atomTypeB;
    float srvDist = 0.0f;
    float alignerDistAB = 0.0f;
    float alignerDistBA = 0.0f;
    float alignerDist = 0.0f;       // max(alignerDistAB, alignerDistBA)
    std::string alignerDir = "A->B"; // "A->B", "B->A", or "equal"
    SubsetAlignmentResult alignAB;
    SubsetAlignmentResult alignBA;
};

struct AtomicClassifyStats
{
    size_t totalAtomicClouds = 0;
    size_t validClouds = 0;
    size_t afterSrv = 0;
    size_t afterAlign = 0;
    double msIndex = 0.0;
    double msSrv = 0.0;
    double msAlign = 0.0;
    double msTotal = 0.0;
    int threads = 1;
};

class AtomicClassifier
{
public:
    static constexpr size_t MAXN = AtomicBottleneckEvaluator::MAX_VERTICES;

    template <typename AtomicCloudVec>
    std::vector<AtomicClassifyMatch> classifyAll(const AtomicCloudVec &atomicEntries,
                                                 const AtomicClassifyParams &p,
                                                 AtomicClassifyStats &st) const
    {
        std::vector<AtomicClassifyMatch> out;
        st = AtomicClassifyStats{};
        st.totalAtomicClouds = atomicEntries.size();

        const int nt = p.parallel ? (p.numThreads > 0 ? p.numThreads : omp_get_max_threads()) : 1;
        st.threads = nt;

        if (atomicEntries.size() < 2)
            return out;

        const double tAll = omp_get_wtime();
        const float eps = p.epsilon;
        auto lim = [&](float factor)
        { return factor * eps * (1.0f + p.relTol) + 1e-6f; };
        const float rSrv = lim(p.srvFactor);
        const float rAlign = lim(p.alignFactor);

        std::cout << "\n[AtomicClassifier] Starting bidirectional batch classification across " << atomicEntries.size()
                  << " atomic clouds\n"
                  << "                   (N=" << p.N << ", eps=" << eps
                  << ", considerSame=" << (p.considerSame ? "true" : "false")
                  << ", optimal=" << (p.optimal ? "true" : "false")
                  << ", threads=" << nt << ")...\n" << std::flush;

        // ---- Stage 0: Build contiguous sorted index by d0 = sortedDistances[0]
        double t0 = omp_get_wtime();
        struct SortedEntry
        {
            size_t cloudIdx;
            float d0;
            uint16_t typeId;
            uint16_t len;
        };

        std::vector<std::string> uniqueTypes;
        auto getTypeId = [&](const std::string &sym) -> uint16_t
        {
            for (size_t i = 0; i < uniqueTypes.size(); ++i)
                if (uniqueTypes[i] == sym)
                    return static_cast<uint16_t>(i);
            uniqueTypes.push_back(sym);
            return static_cast<uint16_t>(uniqueTypes.size() - 1);
        };

        std::vector<SortedEntry> sorted;
        sorted.reserve(atomicEntries.size());
        size_t maxLen = 0;

        for (size_t i = 0; i < atomicEntries.size(); ++i)
        {
            const auto &ac = atomicEntries[i];
            if (!ac.descriptorsComputed || ac.sortedDistances.empty())
                continue;
            uint16_t tId = getTypeId(ac.symbol);
            size_t effectiveLen = (p.N > 0) ? std::min(p.N, ac.sortedDistances.size()) : ac.sortedDistances.size();
            if (effectiveLen == 0)
                continue;
            if (effectiveLen > maxLen)
                maxLen = effectiveLen;

            sorted.push_back({i, ac.sortedDistances[0], tId, static_cast<uint16_t>(effectiveLen)});
        }

        st.validClouds = sorted.size();
        if (sorted.size() < 2)
        {
            st.msTotal = (omp_get_wtime() - tAll) * 1e3;
            return out;
        }

        std::sort(sorted.begin(), sorted.end(), [](const SortedEntry &a, const SortedEntry &b)
                  {
                      if (a.d0 != b.d0)
                          return a.d0 < b.d0;
                      return a.cloudIdx < b.cloudIdx;
                  });

        const size_t M = sorted.size();
        const size_t stride = maxLen;
        std::vector<size_t> cloudIds(M);
        std::vector<float> d0Arr(M);
        std::vector<uint16_t> typeIds(M);
        std::vector<uint16_t> lengths(M);
        std::vector<float> flatDist(M * stride, 0.0f);

        for (size_t row = 0; row < M; ++row)
        {
            size_t cIdx = sorted[row].cloudIdx;
            cloudIds[row] = cIdx;
            d0Arr[row] = sorted[row].d0;
            typeIds[row] = sorted[row].typeId;
            lengths[row] = sorted[row].len;
            const auto &sd = atomicEntries[cIdx].sortedDistances;
            std::copy(sd.begin(), sd.begin() + sorted[row].len, flatDist.begin() + row * stride);
        }
        st.msIndex = (omp_get_wtime() - t0) * 1e3;

        // ---- Stage 1: Sorted sliding-window L_inf range search
        t0 = omp_get_wtime();

        struct SrvPairCand
        {
            size_t idxA;
            size_t idxB;
            float srvDist;
        };

        std::vector<std::vector<SrvPairCand>> threadSrvCands(nt);
        const bool considerSame = p.considerSame;
        std::atomic<size_t> processedSrv(0);
        const size_t srvStep = std::max<size_t>(1, M / 100); // 1% updates

#pragma omp parallel num_threads(nt)
        {
            int tid = omp_get_thread_num();
            auto &localOut = threadSrvCands[tid];

#pragma omp for schedule(dynamic, 64)
            for (long long i = 0; i < static_cast<long long>(M); ++i)
            {
                const float d0A = d0Arr[i];
                const float maxD0 = d0A + rSrv;
                const uint16_t typeA = typeIds[i];
                const uint16_t lenA = lengths[i];
                const float *vecA = flatDist.data() + static_cast<size_t>(i) * stride;
                const size_t idA = cloudIds[i];

                for (size_t j = static_cast<size_t>(i) + 1; j < M; ++j)
                {
                    if (d0Arr[j] > maxD0)
                        break;

                    if (!considerSame && typeIds[j] == typeA)
                        continue;

                    const size_t limit = std::min(lenA, lengths[j]);
                    const float *vecB = flatDist.data() + j * stride;
                    const float dSrv = linfWithin(vecA, vecB, limit, rSrv);
                    if (dSrv > rSrv)
                        continue;

                    const size_t idB = cloudIds[j];
                    if (idA < idB)
                        localOut.push_back({idA, idB, dSrv});
                    else
                        localOut.push_back({idB, idA, dSrv});
                }

                size_t done = ++processedSrv;
                if (done % srvStep == 0 || done == M)
                {
#pragma omp critical
                    {
                        renderProgress("  [Stage 1/2] Atomic SRV Filter ", done, M, omp_get_wtime() - t0, "clouds");
                    }
                }
            }
        }

        std::vector<SrvPairCand> srvCands;
        size_t totalSrvHits = 0;
        for (const auto &v : threadSrvCands)
            totalSrvHits += v.size();
        srvCands.reserve(totalSrvHits);
        for (auto &v : threadSrvCands)
            srvCands.insert(srvCands.end(), v.begin(), v.end());

        st.afterSrv = srvCands.size();
        st.msSrv = (omp_get_wtime() - t0) * 1e3;

        std::cout << "\r  [Stage 1/2] Atomic SRV Filter : Done! ("
                  << std::fixed << std::setprecision(2) << st.msSrv << " ms) | Surviving Candidates: "
                  << st.afterSrv << " pairs                     \n" << std::flush;

        if (srvCands.empty())
        {
            st.msTotal = (omp_get_wtime() - tAll) * 1e3;
            return out;
        }

        // ---- Stage 2: Bidirectional 3D Subset Alignment (A->B and B->A)
        t0 = omp_get_wtime();
        const size_t totalAlignPairs = srvCands.size();
        std::vector<AtomicClassifyMatch> slots(totalAlignPairs);
        std::vector<char> okAlign(totalAlignPairs, 0);
        std::atomic<size_t> processedAlign(0);
        const size_t alignStep = std::max<size_t>(1, totalAlignPairs / 100); // 1% updates

        auto buildCloud = [&](const auto &ac,
                              std::vector<CloudAlignmentEvaluator::Point3F> &pts,
                              std::vector<std::string> &syms)
        {
            pts.resize(ac.atoms.size());
            if (p.requireSameNeighborSpecies)
                syms.resize(ac.atoms.size());
            for (size_t idx = 0; idx < ac.atoms.size(); ++idx)
            {
                pts[idx] = {static_cast<float>(ac.atoms[idx].x),
                            static_cast<float>(ac.atoms[idx].y),
                            static_cast<float>(ac.atoms[idx].z)};
                if (p.requireSameNeighborSpecies)
                    syms[idx] = ac.atoms[idx].symbol;
            }
        };

        auto evalDirected = [&](const auto &srcAc,
                                const std::vector<CloudAlignmentEvaluator::Point3F> &srcCloud,
                                const std::vector<std::string> &srcSyms,
                                const auto &dstAc,
                                const std::vector<CloudAlignmentEvaluator::Point3F> &dstCloud,
                                const std::vector<std::string> &dstSyms,
                                SubsetAlignmentResult &outRes,
                                float &outDist) -> bool
        {
            size_t limitSrc = (p.N > 0) ? std::min(p.N, srcAc.sortedAtomIndices.size()) : srcAc.sortedAtomIndices.size();
            const std::vector<size_t> &idxDstSubset = dstAc.sortedAtomIndices;

            if (limitSrc == 0 || idxDstSubset.empty())
                return false;

            if (idxDstSubset.size() < limitSrc)
                limitSrc = idxDstSubset.size();

            std::vector<size_t> idxSrcSubset(srcAc.sortedAtomIndices.begin(),
                                             srcAc.sortedAtomIndices.begin() + limitSrc);

            AlignmentOptions opt;
            opt.exact = p.optimal;
            opt.acceptDistance = p.optimal ? 0.0f : rAlign;
            opt.refineIterations = p.refineIterations;
            opt.requireSameSymbol = p.requireSameNeighborSpecies;
            if (p.requireSameNeighborSpecies)
            {
                opt.symbolsA = &srcSyms;
                opt.symbolsB = &dstSyms;
            }

            SubsetAlignmentResult res = AtomicCloudAlignmentEvaluator::alignSubsetAndEvaluate(
                srcCloud, idxSrcSubset, dstCloud, idxDstSubset, opt);

            if (!res.success)
                return false;

            float dist = res.bestBottleneck;
            const size_t mSub = idxSrcSubset.size();
            const size_t nSub = idxDstSubset.size();
            if (p.optimal && mSub <= MAXN && nSub <= MAXN && dist <= rAlign)
            {
                dist = computeExactInjectiveBottleneck(
                    srcCloud, idxSrcSubset, srcSyms,
                    dstCloud, idxDstSubset, dstSyms,
                    res.rotation, p.requireSameNeighborSpecies);
                res.bestBottleneck = dist;
            }

            outRes = std::move(res);
            outDist = dist;
            return true;
        };

        auto workerAlign = [&](long long k)
        {
            const auto &cand = srvCands[k];
            const auto &acA = atomicEntries[cand.idxA];
            const auto &acB = atomicEntries[cand.idxB];

            std::vector<CloudAlignmentEvaluator::Point3F> cloudA, cloudB;
            std::vector<std::string> symbolsA, symbolsB;
            buildCloud(acA, cloudA, symbolsA);
            buildCloud(acB, cloudB, symbolsB);

            SubsetAlignmentResult resAB, resBA;
            float distAB = std::numeric_limits<float>::infinity();
            float distBA = std::numeric_limits<float>::infinity();

            bool okAB = evalDirected(acA, cloudA, symbolsA, acB, cloudB, symbolsB, resAB, distAB);
            bool okBA = evalDirected(acB, cloudB, symbolsB, acA, cloudA, symbolsA, resBA, distBA);

            if (okAB && okBA)
            {
                const float maxDist = std::max(distAB, distBA);
                if (maxDist <= rAlign)
                {
                    AtomicClassifyMatch &m = slots[k];
                    m.atomicIdxA = cand.idxA;
                    m.atomicIdxB = cand.idxB;
                    m.datasetIdxA = acA.parentMoleculeIdx;
                    m.atomIdxA = acA.atomIndex;
                    m.atomTypeA = acA.symbol;
                    m.datasetIdxB = acB.parentMoleculeIdx;
                    m.atomIdxB = acB.atomIndex;
                    m.atomTypeB = acB.symbol;
                    m.srvDist = cand.srvDist;
                    m.alignerDistAB = distAB;
                    m.alignerDistBA = distBA;
                    m.alignerDist = maxDist;

                    if (distAB > distBA)
                        m.alignerDir = "A->B";
                    else if (distBA > distAB)
                        m.alignerDir = "B->A";
                    else
                        m.alignerDir = "equal";

                    m.alignAB = std::move(resAB);
                    m.alignBA = std::move(resBA);
                    okAlign[k] = 1;
                }
            }

            size_t done = ++processedAlign;
            if (done % alignStep == 0 || done == totalAlignPairs)
            {
#pragma omp critical
                {
                    renderProgress("  [Stage 2/2] Bidir 3D Aligner  ", done, totalAlignPairs, omp_get_wtime() - t0, "pairs");
                }
            }
        };

        if (p.parallel && totalAlignPairs > 4 && nt > 1)
        {
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
            for (long long k = 0; k < static_cast<long long>(totalAlignPairs); ++k)
                workerAlign(k);
        }
        else
        {
            for (size_t k = 0; k < totalAlignPairs; ++k)
                workerAlign(static_cast<long long>(k));
        }

        for (size_t k = 0; k < totalAlignPairs; ++k)
        {
            if (okAlign[k])
                out.push_back(std::move(slots[k]));
        }

        std::sort(out.begin(), out.end(), [](const AtomicClassifyMatch &a, const AtomicClassifyMatch &b)
                  { return a.alignerDist < b.alignerDist; });

        if (p.maxMatches > 0 && out.size() > p.maxMatches)
            out.resize(p.maxMatches);

        st.afterAlign = out.size();
        st.msAlign = (omp_get_wtime() - t0) * 1e3;
        st.msTotal = (omp_get_wtime() - tAll) * 1e3;

        std::cout << "\r  [Stage 2/2] Bidir 3D Aligner  : Done! ("
                  << std::fixed << std::setprecision(2) << st.msAlign << " ms) | Matched Close Pairs: "
                  << st.afterAlign << "                         \n" << std::flush;

        return out;
    }

    static bool writeCsv(const std::string &csvPath, const std::vector<AtomicClassifyMatch> &matches)
    {
        if (csvPath.empty())
            return false;

        std::ofstream csv(csvPath);
        if (!csv.is_open())
        {
            std::cerr << "[AtomicClassifier] Error: Could not open CSV file for writing: " << csvPath << "\n";
            return false;
        }

        csv << "DatasetIdxA,AtomIdxA,AtomTypeA,DatasetIdxB,AtomIdxB,AtomTypeB,SRVDist,AlignerDist,AlignerDir\n";
        for (const auto &m : matches)
        {
            csv << m.datasetIdxA << ","
                << m.atomIdxA << ","
                << m.atomTypeA << ","
                << m.datasetIdxB << ","
                << m.atomIdxB << ","
                << m.atomTypeB << ","
                << std::fixed << std::setprecision(6)
                << m.srvDist << ","
                << m.alignerDist << ","
                << m.alignerDir << "\n";
        }
        std::cout << "[AtomicClassifier] Successfully wrote " << matches.size()
                  << " atomic pairs to " << csvPath << "\n";
        return true;
    }

    static void printBatchReport(const AtomicClassifyParams &p,
                                 const AtomicClassifyStats &s,
                                 const std::vector<AtomicClassifyMatch> &m,
                                 const std::string &csvPath)
    {
        std::cout << "\n=======================================================\n"
                  << "  ALL-ATOMS CLASSIFICATION BATCH SUMMARY\n"
                  << "=======================================================\n"
                  << std::fixed << std::setprecision(2)
                  << "  Neighbors (N)       : " << p.N << "\n"
                  << "  Epsilon Threshold   : " << p.epsilon << "\n"
                  << "  ConsiderSame        : " << (p.considerSame ? "true" : "false") << "\n"
                  << "  Optimal Aligner     : " << (p.optimal ? "true" : "false") << "\n"
                  << "  Valid Atomic Clouds : " << s.validClouds << " / " << s.totalAtomicClouds << "\n"
                  << "  Total Time Elapsed  : " << s.msTotal << " ms\n"
                  << "  Threads Used        : " << s.threads << "\n"
                  << "  CSV Export Path     : " << (csvPath.empty() ? "(none)" : csvPath) << "\n"
                  << "-------------------------------------------------------\n"
                  << "  STAGE EXECUTION STATS & TIMINGS:\n"
                  << "    0. Index Build      : " << std::setw(8) << s.msIndex << " ms\n"
                  << "    1. Atomic SRV Range : " << std::setw(8) << s.afterSrv << " pairs | "
                  << std::setw(8) << s.msSrv << " ms\n"
                  << "    2. Bidir Aligner    : " << std::setw(8) << s.afterAlign << " pairs | "
                  << std::setw(8) << s.msAlign << " ms\n"
                  << "-------------------------------------------------------\n";
        if (m.empty())
        {
            std::cout << "  No atomic neighborhoods found within epsilon threshold.\n";
        }
        else
        {
            std::cout << "  TOP CLOSE ATOMIC NEIGHBORHOODS (Sorted by max(A->B, B->A)):\n"
                      << "  " << std::left << std::setw(8) << "MolA"
                      << std::setw(8) << "AtomA"
                      << std::setw(7) << "TypeA"
                      << std::setw(8) << "MolB"
                      << std::setw(8) << "AtomB"
                      << std::setw(7) << "TypeB"
                      << std::setw(11) << "SRVDist"
                      << std::setw(12) << "AlignDist"
                      << "Dir\n";
            size_t lim = std::min<size_t>(m.size(), 20);
            for (size_t i = 0; i < lim; ++i)
            {
                std::cout << "  " << std::left << std::setw(8) << m[i].datasetIdxA
                          << std::setw(8) << m[i].atomIdxA
                          << std::setw(7) << m[i].atomTypeA
                          << std::setw(8) << m[i].datasetIdxB
                          << std::setw(8) << m[i].atomIdxB
                          << std::setw(7) << m[i].atomTypeB
                          << std::setprecision(6)
                          << std::setw(11) << m[i].srvDist
                          << std::setw(12) << m[i].alignerDist
                          << m[i].alignerDir << "\n";
            }
            if (m.size() > lim)
                std::cout << "  ... " << (m.size() - lim) << " more pairs\n";
        }
        std::cout << "=======================================================\n\n";
    }

private:
    static void renderProgress(const std::string &prefix, size_t done, size_t total, double elapsedSec, const std::string &unitName, int barWidth = 22)
    {
        if (total == 0) return;
        float frac = std::min(1.0f, static_cast<float>(done) / static_cast<float>(total));
        int filled = static_cast<int>(frac * barWidth);

        std::string bar = "[";
        for (int i = 0; i < barWidth; ++i)
        {
            if (i < filled) bar += "=";
            else if (i == filled) bar += ">";
            else bar += " ";
        }
        bar += "]";

        double rate = (elapsedSec > 0.001) ? (static_cast<double>(done) / elapsedSec) : 0.0;
        double remSec = (rate > 0.0) ? (static_cast<double>(total - done) / rate) : 0.0;

        std::string rateStr;
        if (rate >= 1000.0)
            rateStr = std::to_string(static_cast<int>(rate / 1000.0)) + "k " + unitName + "/s";
        else
            rateStr = std::to_string(static_cast<int>(rate)) + " " + unitName + "/s";

        int remMin = static_cast<int>(remSec) / 60;
        int remSecMod = static_cast<int>(remSec) % 60;
        std::ostringstream etaOss;
        etaOss << std::setfill('0') << std::setw(2) << remMin << ":" << std::setw(2) << remSecMod;

        std::cout << "\r" << prefix << ": " << bar << " "
                  << std::fixed << std::setprecision(1) << std::setw(5) << (frac * 100.0f) << "% "
                  << "(" << done << "/" << total << ", " << rateStr;

        if (done < total && rate > 0.0)
            std::cout << ", ETA: " << etaOss.str();

        std::cout << ")   " << std::flush;
    }

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

    static float computeExactInjectiveBottleneck(
        const std::vector<CloudAlignmentEvaluator::Point3F> &cloudA,
        const std::vector<size_t> &idxA_subset,
        const std::vector<std::string> &symbolsA,
        const std::vector<CloudAlignmentEvaluator::Point3F> &cloudB,
        const std::vector<size_t> &idxB_subset,
        const std::vector<std::string> &symbolsB,
        const float *R,
        bool requireSameSymbol)
    {
        const size_t m = idxA_subset.size();
        const size_t n = idxB_subset.size();
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
                if (requireSameSymbol && symbolsA[idxA_subset[i]] != symbolsB[idxB_subset[j]])
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
            return AtomicBottleneckEvaluator::bottleneckInjective(cost.data(), m, n, lo, hi);
        return 0.0f;
    }
};