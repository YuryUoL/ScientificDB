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
#include <omp.h>
#include "FaissIndexManager.h"
#include "Aligners/CloudAlignmentEvaluator.h"
#include "Aligners/BottleneckEvaluator.h"

struct DupCheckParams
{
    float epsilon = 0.05f;              // Max per-atom displacement after alignment (Angstroms)
    float srvFactor = 1.0f;             // Stage 1 radius factor
    float sdvFactor = 2.0f;             // Stage 2 radius factor
    float pddFactor = 2.0f;             // Stage 3 radius factor
    float alignFactor = 1.0f;           // Stage 4 aligner radius factor
    float relTol = 1e-4f;               // Float-noise slack on thresholds
    bool requireSameComposition = true; // Candidates must share element symbol multiset
    bool requireSameLabel = false;      // Candidates must share the same label/comment line
    bool rejectSpeciesMismatch = false; // Drop matches whose aligner pairs different elements
    bool exactDistance = true;          // True: finds true minimum (<= eps); false: stops at first <= eps
    int refineIterations = 0;           // Aligner Kabsch refinement rounds (0 = disabled by default)
    size_t maxMatches = 0;              // Keep only best N matches (0 = all)
    int numThreads = 0;                 // 0 = OpenMP default
    bool parallel = true;               // Enable/disable multi-threaded candidate evaluation

    bool set(const std::string &k, const std::string &v)
    {
        try
        {
            auto b = [&]()
            { return v == "1" || v == "on" || v == "true"; };
            if (k == "epsilon" || k == "eps")
                epsilon = std::stof(v);
            else if (k == "srvf")
                srvFactor = std::stof(v);
            else if (k == "sdvf")
                sdvFactor = std::stof(v);
            else if (k == "pddf")
                pddFactor = std::stof(v);
            else if (k == "alignf")
                alignFactor = std::stof(v);
            else if (k == "tol")
                relTol = std::stof(v);
            else if (k == "comp")
                requireSameComposition = b();
            else if (k == "label")
                requireSameLabel = b();
            else if (k == "species")
                rejectSpeciesMismatch = b();
            else if (k == "exact")
                exactDistance = b();
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

struct DupMatch
{
    size_t queryIdx = 0;
    size_t matchIdx = 0;
    size_t numAtoms = 0;
    float srvLinf = 0.0f;
    float sdvLinf = 0.0f;
    float pddBottleneck = 0.0f;
    float alignBottleneck = 0.0f;
    int speciesMismatches = 0;
    AlignmentResult align;
};

struct DupRunStats
{
    size_t numAtoms = 0, groupSize = 0;
    size_t afterSrv = 0, afterSdv = 0, afterPdd = 0, afterAlign = 0;
    double msSrv = 0, msSdv = 0, msPdd = 0, msAlign = 0, msTotal = 0;
    int threads = 1;
    bool verified = true;

    // Batch aggregate stats
    size_t batchSrvCount = 0;
    size_t batchSdvCount = 0;
    size_t batchPddCount = 0;
    size_t batchAlignCount = 0;
    double batchMsSrv = 0.0;
    double batchMsSdv = 0.0;
    double batchMsPdd = 0.0;
    double batchMsAlign = 0.0;
};

struct DefaultGetLabel
{
    std::string operator()(size_t) const { return ""; }
};

class DuplicateChecker
{
public:
    static constexpr size_t MAXN = BottleneckEvaluator::MAX_VERTICES;

    template <class GetAtoms, class GetLabel = DefaultGetLabel>
    std::vector<DupMatch> run(const FaissIndexManager &fm, size_t queryIdx,
                              const MolecularDescriptors &q,
                              const std::vector<AtomRecord> &qAtoms,
                              GetAtoms getAtoms, GetLabel getLabel, const DupCheckParams &p,
                              DupRunStats &st) const
    {
        std::vector<DupMatch> out;
        st = DupRunStats{};
        const size_t n = q.numAtoms;
        st.numAtoms = n;
        st.groupSize = fm.groupSize(n);
        st.threads = p.numThreads > 0 ? p.numThreads : omp_get_max_threads();
        if (n < 2)
            return out;

        const double tAll = omp_get_wtime();
        const float eps = p.epsilon;
        auto lim = [&](float factor)
        { return factor * eps * (1.0f + p.relTol) + 1e-6f; };
        const int nt = st.threads;

        std::cout << "[DupCheck] Starting single-molecule duplicate check for query index " << queryIdx 
                  << " (Atoms: " << n << ", Database Group Size: " << st.groupSize << ")\n" << std::flush;

        // ---- Fused Stage 1 (SRV LInf) + Stage 2 (Composition + SDV LInf)
        std::cout << "  [Stage 1+2/4] Running Fused SRV + SDV Range Search (threads: " << nt << ")... " << std::flush;
        double t0 = omp_get_wtime();
        std::vector<SrvSdvCandidate> stage2Cands;
        fm.rangeSearchSrvSdvSingle(queryIdx, lim(p.srvFactor), lim(p.sdvFactor),
                                   p.requireSameComposition, nt, p.parallel,
                                   st.afterSrv, stage2Cands);

        if (p.requireSameLabel && !stage2Cands.empty())
        {
            const std::string qLabel = getLabel(queryIdx);
            std::vector<SrvSdvCandidate> filtered;
            filtered.reserve(stage2Cands.size());
            for (const auto &c : stage2Cands)
            {
                if (getLabel(c.idxB) == qLabel)
                    filtered.push_back(c);
            }
            stage2Cands = std::move(filtered);
        }

        st.afterSdv = stage2Cands.size();
        st.msSdv = (omp_get_wtime() - t0) * 1e3;
        std::cout << "Done! (" << std::fixed << std::setprecision(2) << st.msSdv << " ms) | Candidates surviving: " << st.afterSdv << "\n" << std::flush;

        if (n > MAXN)
        {
            std::cout << "  [Warning] Molecule atom count (" << n << ") exceeds MAXN (" << MAXN << "). PDD/aligner skipped.\n" << std::flush;
            st.verified = false;
            st.msTotal = (omp_get_wtime() - tAll) * 1e3;
            return out;
        }

        // ---- Stage 3: PDD (Row-wise LInf bottleneck matching via BottleneckEvaluator)
        t0 = omp_get_wtime();
        struct Cand
        {
            size_t g;
            float srv, sdv, pdd;
        };
        std::vector<Cand> cands(stage2Cands.size());
        for (size_t i = 0; i < stage2Cands.size(); ++i)
        {
            cands[i] = {stage2Cands[i].idxB, stage2Cands[i].srvLinf, stage2Cands[i].sdvLinf, 0.0f};
        }

        {
            const float pddLim = lim(p.pddFactor);
            std::vector<char> keep(cands.size(), 0);
            std::atomic<size_t> processed(0);
            const size_t totalCands = cands.size();

            std::cout << "  [Stage 3/4] Running PDD Bottleneck Evaluation (0 / " << totalCands << " completed, 0%)... " << std::flush;

            auto worker3 = [&](long long i)
            {
                const MolecularDescriptors *d = fm.descriptor(cands[i].g);
                float v = 0.f;
                if (d && pddCheck(q.pdd, d->pdd, n, pddLim, v))
                {
                    cands[i].pdd = v;
                    keep[i] = 1;
                }
                size_t done = ++processed;
                if (totalCands > 100 && (done % (totalCands / 20 + 1) == 0 || done == totalCands))
                {
                    #pragma omp critical
                    {
                        std::cout << "\r  [Stage 3/4] Running PDD Bottleneck Evaluation (" << done << " / " << totalCands 
                                  << " completed, " << (done * 100 / totalCands) << "%)... " << std::flush;
                    }
                }
            };

            if (p.parallel && cands.size() > 4 && nt > 1)
            {
#pragma omp parallel for schedule(dynamic, 4) num_threads(nt)
                for (long long i = 0; i < static_cast<long long>(cands.size()); ++i)
                    worker3(i);
            }
            else
            {
                for (size_t i = 0; i < cands.size(); ++i)
                    worker3(static_cast<long long>(i));
            }
            compact(cands, keep);
        }
        st.afterPdd = cands.size();
        st.msPdd = (omp_get_wtime() - t0) * 1e3;
        std::cout << "\r  [Stage 3/4] PDD Bottleneck Evaluation Complete! (" << std::fixed << std::setprecision(2) << st.msPdd << " ms) | Survivors: " << st.afterPdd << " / " << st.batchPddCount << "               \n" << std::flush;

        // ---- Stage 4: Exact Aligner
        t0 = omp_get_wtime();
        {
            std::vector<CloudAlignmentEvaluator::Point3F> cloudQ = toCloud(qAtoms);
            AlignmentOptions opt;
            opt.exact = p.exactDistance;
            opt.acceptDistance = p.exactDistance ? 0.0f : lim(p.alignFactor);
            opt.refineIterations = p.refineIterations;
            const float accept = lim(p.alignFactor);

            std::vector<DupMatch> slots(cands.size());
            std::vector<char> ok(cands.size(), 0);
            std::atomic<size_t> processed(0);
            const size_t totalCands = cands.size();

            std::cout << "  [Stage 4/4] Running Exact 3D Kabsch & Aligner (0 / " << totalCands << " completed, 0%)... " << std::flush;

            auto worker4 = [&](long long i)
            {
                const auto &cAtoms = getAtoms(cands[i].g);
                AlignmentResult res = CloudAlignmentEvaluator::alignAndEvaluate(cloudQ, toCloud(cAtoms), opt);
                if (res.success && res.bestBottleneck <= accept)
                {
                    int mism = 0;
                    for (size_t a = 0; a < res.matching.size() && a < qAtoms.size(); ++a)
                        if (std::string(qAtoms[a].symbol) != std::string(cAtoms[res.matching[a]].symbol))
                            ++mism;
                    if (!p.rejectSpeciesMismatch || mism == 0)
                    {
                        DupMatch &m = slots[i];
                        m.queryIdx = queryIdx;
                        m.matchIdx = cands[i].g;
                        m.numAtoms = n;
                        m.srvLinf = cands[i].srv;
                        m.sdvLinf = cands[i].sdv;
                        m.pddBottleneck = cands[i].pdd;
                        m.alignBottleneck = res.bestBottleneck;
                        m.speciesMismatches = mism;
                        m.align = std::move(res);
                        ok[i] = 1;
                    }
                }

                size_t done = ++processed;
                if (totalCands > 20 && (done % (totalCands / 20 + 1) == 0 || done == totalCands))
                {
                    #pragma omp critical
                    {
                        std::cout << "\r  [Stage 4/4] Running Exact 3D Kabsch & Aligner (" << done << " / " << totalCands 
                                  << " completed, " << (done * 100 / totalCands) << "%)... " << std::flush;
                    }
                }
            };

            if (p.parallel && cands.size() > 4 && nt > 1)
            {
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
                for (long long i = 0; i < static_cast<long long>(cands.size()); ++i)
                    worker4(i);
            }
            else
            {
                for (size_t i = 0; i < cands.size(); ++i)
                    worker4(static_cast<long long>(i));
            }

            for (size_t i = 0; i < slots.size(); ++i)
                if (ok[i])
                    out.push_back(std::move(slots[i]));
        }
        std::sort(out.begin(), out.end(), [](const DupMatch &a, const DupMatch &b)
                  { return a.alignBottleneck < b.alignBottleneck; });
        if (p.maxMatches > 0 && out.size() > p.maxMatches)
            out.resize(p.maxMatches);
        st.afterAlign = out.size();
        st.msAlign = (omp_get_wtime() - t0) * 1e3;
        st.msTotal = (omp_get_wtime() - tAll) * 1e3;
        
        std::cout << "\r  [Stage 4/4] Exact Aligner Complete! (" << std::fixed << std::setprecision(2) << st.msAlign << " ms) | Verified Duplicates Found: " << st.afterAlign << "               \n" << std::flush;
        return out;
    }

    // Batch runner for ALL indices using upper-triangular sorted-bucket Stage 1+2
    template <class GetAtoms, class GetLabel = DefaultGetLabel>
    std::vector<DupMatch> runAll(const FaissIndexManager &fm, size_t totalMols,
                                 GetAtoms getAtoms, GetLabel getLabel, const DupCheckParams &p,
                                 DupRunStats &stTotal) const
    {
        std::vector<DupMatch> allMatches;
        stTotal = DupRunStats{};
        const int nt = p.numThreads > 0 ? p.numThreads : omp_get_max_threads();
        stTotal.threads = nt;

        if (totalMols < 2)
            return allMatches;

        const double tStart = omp_get_wtime();
        const float eps = p.epsilon;
        auto lim = [&](float factor)
        { return factor * eps * (1.0f + p.relTol) + 1e-6f; };

        std::cout << "[DupCheck] Starting BATCH duplicate check across all " << totalMols 
                  << " molecules (threads: " << nt << ")...\n" << std::flush;

        // ---- Fused Stage 1 + Stage 2 across all sorted buckets (upper-triangular j > i only)
        std::cout << "  [Stage 1+2/4] Running All-Pairs Fused SRV + SDV Range Search... " << std::flush;
        double t0 = omp_get_wtime();
        size_t totalSrvSurvPairs = 0;
        std::vector<SrvSdvCandidate> stage2Pairs;
        fm.rangeSearchSrvSdvAllPairs(lim(p.srvFactor), lim(p.sdvFactor),
                                     p.requireSameComposition, nt, p.parallel,
                                     totalSrvSurvPairs, stage2Pairs);

        if (p.requireSameLabel && !stage2Pairs.empty())
        {
            std::vector<SrvSdvCandidate> filtered;
            filtered.reserve(stage2Pairs.size());
            for (const auto &c : stage2Pairs)
            {
                if (getLabel(c.idxA) == getLabel(c.idxB))
                    filtered.push_back(c);
            }
            stage2Pairs = std::move(filtered);
        }

        stTotal.batchSrvCount = totalMols;
        stTotal.batchSdvCount = totalSrvSurvPairs;
        stTotal.batchMsSdv = (omp_get_wtime() - t0) * 1e3;
        std::cout << "Done! (" << std::fixed << std::setprecision(2) << stTotal.batchMsSdv << " ms) | Surviving Pairs: " << stage2Pairs.size() << "\n" << std::flush;

        // ---- Stage 3: PDD Bottleneck matching on surviving pairs
        t0 = omp_get_wtime();
        stTotal.batchPddCount = stage2Pairs.size();
        struct PairCand
        {
            size_t idxA, idxB, numAtoms;
            float srv, sdv, pdd;
        };
        std::vector<PairCand> pddCands(stage2Pairs.size());
        std::vector<char> keepPdd(stage2Pairs.size(), 0);
        const float pddLim = lim(p.pddFactor);
        std::atomic<size_t> processedPdd(0);
        const size_t totalPddPairs = stage2Pairs.size();

        std::cout << "  [Stage 3/4] Running Batch PDD Bottleneck Evaluation (0 / " << totalPddPairs << " completed, 0%)... " << std::flush;

        auto workerPdd = [&](long long k)
        {
            const auto &c = stage2Pairs[k];
            const MolecularDescriptors *dA = fm.descriptor(c.idxA);
            const MolecularDescriptors *dB = fm.descriptor(c.idxB);
            if (dA && dB && dA->numAtoms == dB->numAtoms)
            {
                const size_t n = dA->numAtoms;
                if (n <= MAXN)
                {
                    float v = 0.0f;
                    if (pddCheck(dA->pdd, dB->pdd, n, pddLim, v))
                    {
                        pddCands[k] = {c.idxA, c.idxB, n, c.srvLinf, c.sdvLinf, v};
                        keepPdd[k] = 1;
                    }
                }
            }

            size_t done = ++processedPdd;
            if (totalPddPairs > 50 && (done % (totalPddPairs / 20 + 1) == 0 || done == totalPddPairs))
            {
                #pragma omp critical
                {
                    std::cout << "\r  [Stage 3/4] Running Batch PDD Bottleneck Evaluation (" << done << " / " << totalPddPairs 
                              << " completed, " << (done * 100 / totalPddPairs) << "%)... " << std::flush;
                }
            }
        };

        if (p.parallel && stage2Pairs.size() > 4 && nt > 1)
        {
#pragma omp parallel for schedule(dynamic, 16) num_threads(nt)
            for (long long k = 0; k < static_cast<long long>(stage2Pairs.size()); ++k)
                workerPdd(k);
        }
        else
        {
            for (size_t k = 0; k < stage2Pairs.size(); ++k)
                workerPdd(static_cast<long long>(k));
        }
        compact(pddCands, keepPdd);
        stTotal.batchMsPdd = (omp_get_wtime() - t0) * 1e3;
        std::cout << "\r  [Stage 3/4] Batch PDD Evaluation Complete! (" << std::fixed << std::setprecision(2) << stTotal.batchMsPdd << " ms) | PDD Survivors: " << pddCands.size() << " / " << totalPddPairs << "               \n" << std::flush;

        // ---- Stage 4: Exact 3D Kabsch + Bottleneck Alignment on PDD survivors
        t0 = omp_get_wtime();
        stTotal.batchAlignCount = pddCands.size();

        if (!pddCands.empty())
        {
            AlignmentOptions opt;
            opt.exact = p.exactDistance;
            opt.acceptDistance = p.exactDistance ? 0.0f : lim(p.alignFactor);
            opt.refineIterations = p.refineIterations;
            const float accept = lim(p.alignFactor);

            std::vector<DupMatch> slots(pddCands.size());
            std::vector<char> okAlign(pddCands.size(), 0);
            std::atomic<size_t> processedAlign(0);
            const size_t totalAlignPairs = pddCands.size();

            std::cout << "  [Stage 4/4] Running Batch Exact 3D Kabsch & Aligner (0 / " << totalAlignPairs << " completed, 0%)... " << std::flush;

            auto workerAlign = [&](long long k)
            {
                const auto &c = pddCands[k];
                const auto &atomsA = getAtoms(c.idxA);
                const auto &atomsB = getAtoms(c.idxB);

                AlignmentResult res = CloudAlignmentEvaluator::alignAndEvaluate(toCloud(atomsA), toCloud(atomsB), opt);
                if (res.success && res.bestBottleneck <= accept)
                {
                    int mism = 0;
                    for (size_t a = 0; a < res.matching.size() && a < atomsA.size(); ++a)
                        if (std::string(atomsA[a].symbol) != std::string(atomsB[res.matching[a]].symbol))
                            ++mism;
                    if (!p.rejectSpeciesMismatch || mism == 0)
                    {
                        DupMatch &m = slots[k];
                        m.queryIdx = c.idxA;
                        m.matchIdx = c.idxB;
                        m.numAtoms = c.numAtoms;
                        m.srvLinf = c.srv;
                        m.sdvLinf = c.sdv;
                        m.pddBottleneck = c.pdd;
                        m.alignBottleneck = res.bestBottleneck;
                        m.speciesMismatches = mism;
                        m.align = std::move(res);
                        okAlign[k] = 1;
                    }
                }

                size_t done = ++processedAlign;
                if (totalAlignPairs > 20 && (done % (totalAlignPairs / 20 + 1) == 0 || done == totalAlignPairs))
                {
                    #pragma omp critical
                    {
                        std::cout << "\r  [Stage 4/4] Running Batch Exact 3D Kabsch & Aligner (" << done << " / " << totalAlignPairs 
                                  << " completed, " << (done * 100 / totalAlignPairs) << "%)... " << std::flush;
                    }
                }
            };

            if (p.parallel && pddCands.size() > 4 && nt > 1)
            {
#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
                for (long long k = 0; k < static_cast<long long>(pddCands.size()); ++k)
                    workerAlign(k);
            }
            else
            {
                for (size_t k = 0; k < pddCands.size(); ++k)
                    workerAlign(static_cast<long long>(k));
            }

            for (size_t k = 0; k < slots.size(); ++k)
            {
                if (okAlign[k])
                    allMatches.push_back(std::move(slots[k]));
            }
        }
        stTotal.batchMsAlign = (omp_get_wtime() - t0) * 1e3;

        std::sort(allMatches.begin(), allMatches.end(), [](const DupMatch &a, const DupMatch &b)
                  { return a.alignBottleneck < b.alignBottleneck; });

        if (p.maxMatches > 0 && allMatches.size() > p.maxMatches)
            allMatches.resize(p.maxMatches);

        stTotal.afterAlign = allMatches.size();
        stTotal.msTotal = (omp_get_wtime() - tStart) * 1e3;

        std::cout << "\r  [Stage 4/4] Batch Exact Aligner Complete! (" << std::fixed << std::setprecision(2) << stTotal.batchMsAlign << " ms) | Verified Duplicate Pairs Found: " << allMatches.size() << "               \n" << std::flush;
        return allMatches;
    }

    // --------------------------------------------------------------- Report
    static void printReport(size_t queryIdx, const DupCheckParams &p, const DupRunStats &s,
                            const std::vector<DupMatch> &m)
    {
        const float e = p.epsilon;
        std::cout << "\n=======================================================\n"
                  << "  DUPLICATE CHECK  query=" << queryIdx << "  eps=" << e << "  atoms=" << s.numAtoms
                  << "  threads=" << s.threads << "\n"
                  << "=======================================================\n"
                  << std::fixed << std::setprecision(2)
                  << "  same-atom-count DB size : " << s.groupSize << "\n"
                  << "  1 SRV  Linf <= " << std::setprecision(4) << p.srvFactor * e << "   -> " << s.afterSrv << "\n"
                  << "  2 SDV  Linf <= " << std::setprecision(4) << p.sdvFactor * e
                  << (p.requireSameComposition ? " +comp" : "")
                  << (p.requireSameLabel ? " +label" : "") << " -> " << s.afterSdv
                  << "   (" << std::setprecision(2) << (s.msSrv + s.msSdv) << " ms fused)\n";
        if (!s.verified)
        {
            std::cout << "  n > " << MAXN << " atoms: PDD/aligner unavailable, " << s.afterSdv
                      << " unverified candidates after stage 2.\n"
                      << "=======================================================\n\n";
            return;
        }
        std::cout << "  3 PDD  Linf <= " << std::setprecision(4) << p.pddFactor * e << "   -> " << s.afterPdd
                  << "   (" << std::setprecision(2) << s.msPdd << " ms)\n"
                  << "  4 aligner   <= " << std::setprecision(4) << e << "   -> " << s.afterAlign
                  << "   (" << std::setprecision(2) << s.msAlign << " ms)\n"
                  << "  total " << s.msTotal << " ms\n"
                  << "-------------------------------------------------------\n";
        if (m.empty())
        {
            std::cout << "  NO DUPLICATE: molecule is unique at eps = " << e << "\n";
        }
        else
        {
            std::cout << "  " << m.size() << " DUPLICATE(S) FOUND:\n"
                      << "  " << std::left << std::setw(8) << "MolA"
                      << std::setw(8) << "MolB"
                      << std::setw(10) << "Atoms"
                      << std::setw(11) << "align"
                      << std::setw(11) << "pdd"
                      << std::setw(11) << "sdv"
                      << std::setw(11) << "srv"
                      << "mismatch\n";
            const size_t lim = std::min<size_t>(m.size(), 15);
            for (size_t i = 0; i < lim; ++i)
                std::cout << "  " << std::left << std::setw(8) << m[i].queryIdx
                          << std::setw(8) << m[i].matchIdx
                          << std::setw(10) << m[i].numAtoms
                          << std::setprecision(6)
                          << std::setw(11) << m[i].alignBottleneck
                          << std::setw(11) << m[i].pddBottleneck
                          << std::setw(11) << m[i].sdvLinf
                          << std::setw(11) << m[i].srvLinf
                          << m[i].speciesMismatches << "\n";
            if (m.size() > lim)
                std::cout << "  ... " << (m.size() - lim) << " more\n";
        }
        std::cout << "=======================================================\n\n";
    }

    static void printBatchReport(const DupCheckParams &p, const DupRunStats &s,
                                 const std::vector<DupMatch> &m, const std::string &csvPath)
    {
        double sumStageMs = s.batchMsSrv + s.batchMsSdv + s.batchMsPdd + s.batchMsAlign;
        if (sumStageMs < 1e-3)
            sumStageMs = 1.0;

        std::cout << "\n=======================================================\n"
                  << "  ALL-INDICES DUPLICATE CHECK BATCH SUMMARY\n"
                  << "=======================================================\n"
                  << std::fixed << std::setprecision(2)
                  << "  Epsilon Threshold   : " << p.epsilon << "\n"
                  << "  Total Checked Pairs : " << m.size() << "\n"
                  << "  Total Time Elapsed  : " << s.msTotal << " ms\n"
                  << "  Threads Used        : " << s.threads << "\n"
                  << "  CSV Export Path     : " << csvPath << "\n"
                  << "-------------------------------------------------------\n"
                  << "  STAGE EXECUTION STATS & TIMINGS:\n"
                  << "    1+2. Fused SRV+SDV    : " << std::setw(8) << s.batchSdvCount << " SRV hits | "
                  << std::setw(8) << (s.batchMsSrv + s.batchMsSdv) << " ms ("
                  << std::setprecision(1) << ((s.batchMsSrv + s.batchMsSdv) / sumStageMs) * 100.0 << "%)\n"
                  << "    3.   PDD Evaluations  : " << std::setw(8) << s.batchPddCount << " pairs    | "
                  << std::setw(8) << s.batchMsPdd << " ms (" << (s.batchMsPdd / sumStageMs) * 100.0 << "%)\n"
                  << "    4.   Aligned Evals    : " << std::setw(8) << s.batchAlignCount << " pairs    | "
                  << std::setw(8) << s.batchMsAlign << " ms (" << (s.batchMsAlign / sumStageMs) * 100.0 << "%)\n"
                  << "-------------------------------------------------------\n";
        if (m.empty())
        {
            std::cout << "  No duplicates found within epsilon threshold.\n";
        }
        else
        {
            std::cout << "  TOP DUPLICATE PAIRS (Sorted by Aligner Distance):\n"
                      << "  " << std::left << std::setw(8) << "MolA"
                      << std::setw(8) << "MolB"
                      << std::setw(10) << "Atoms"
                      << std::setw(11) << "align"
                      << std::setw(11) << "pdd"
                      << std::setw(11) << "sdv"
                      << std::setw(11) << "srv"
                      << "mismatch\n";
            size_t lim = std::min<size_t>(m.size(), 20);
            for (size_t i = 0; i < lim; ++i)
            {
                std::cout << "  " << std::left << std::setw(8) << m[i].queryIdx
                          << std::setw(8) << m[i].matchIdx
                          << std::setw(10) << m[i].numAtoms
                          << std::setprecision(6)
                          << std::setw(11) << m[i].alignBottleneck
                          << std::setw(11) << m[i].pddBottleneck
                          << std::setw(11) << m[i].sdvLinf
                          << std::setw(11) << m[i].srvLinf
                          << m[i].speciesMismatches << "\n";
            }
            if (m.size() > lim)
                std::cout << "  ... " << (m.size() - lim) << " more pairs\n";
        }
        std::cout << "=======================================================\n\n";
    }

private:
    template <class Atoms>
    static std::vector<CloudAlignmentEvaluator::Point3F> toCloud(const Atoms &atoms)
    {
        std::vector<CloudAlignmentEvaluator::Point3F> c(atoms.size());
        for (size_t i = 0; i < atoms.size(); ++i)
            c[i] = {static_cast<float>(atoms[i].x), static_cast<float>(atoms[i].y), static_cast<float>(atoms[i].z)};
        return c;
    }

    template <class T>
    static void compact(std::vector<T> &v, const std::vector<char> &keep)
    {
        size_t w = 0;
        for (size_t r = 0; r < v.size(); ++r)
            if (keep[r])
                v[w++] = std::move(v[r]);
        v.resize(w);
    }

    static bool pddCheck(const std::vector<float> &pa, const std::vector<float> &pb, size_t n,
                         float thr, float &value)
    {
        if (n == 0 || n > MAXN || pa.size() != n * n || pb.size() != n * n)
            return false;

        // 1. Ultra-fast bitmask filter with early exit (rejects ~99.3% of pairs in nanoseconds)
        if (!BottleneckEvaluator::passesPddThreshold(pa.data(), pb.data(), n, thr))
            return false;

        // 2. Exact bottleneck value only computed for the tiny fraction (~0.7%) that survive
        float costMatrix[MAXN * MAXN];
        BottleneckEvaluator::computeCostMatrix(pa.data(), pb.data(), n, costMatrix);
        value = BottleneckEvaluator::computeFromCostMatrix(costMatrix, n, 0.0f, thr);
        return true;
    }
};