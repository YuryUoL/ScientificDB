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
#include "CloudAligmentEvaluator.h"
#include "BottleneckEvaluator.h"

struct DupCheckParams
{
    float epsilon = 0.05f;               // Max per-atom displacement after alignment (Angstroms)
    float srvFactor = 1.0f;              // Stage 1 radius factor[cite: 3]
    float sdvFactor = 2.0f;              // Stage 2 radius factor[cite: 3]
    float pddFactor = 2.0f;              // Stage 3 radius factor[cite: 3]
    float alignFactor = 1.0f;            // Stage 4 aligner radius factor
    float relTol = 1e-4f;                // Float-noise slack on thresholds[cite: 3]
    bool requireSameComposition = true;  // Candidates must share element symbol multiset[cite: 3]
    bool requireSameLabel = false;       // Candidates must share the same label/comment line[cite: 3]
    bool rejectSpeciesMismatch = false;  // Drop matches whose aligner pairs different elements[cite: 3]
    bool exactDistance = true;           // True: finds true minimum (<= eps); false: stops at first <= eps[cite: 3]
    int refineIterations = 0;            // Aligner Kabsch refinement rounds (0 = disabled by default)[cite: 3]
    size_t maxMatches = 0;               // Keep only best N matches (0 = all)[cite: 3]
    int numThreads = 0;                  // 0 = OpenMP default[cite: 3]
    bool parallel = true;                // Enable/disable multi-threaded candidate evaluation[cite: 3]

    bool set(const std::string &k, const std::string &v)
    {
        try
        {
            auto b = [&]() { return v == "1" || v == "on" || v == "true"; };
            if (k == "epsilon" || k == "eps") epsilon = std::stof(v);
            else if (k == "srvf") srvFactor = std::stof(v);
            else if (k == "sdvf") sdvFactor = std::stof(v);
            else if (k == "pddf") pddFactor = std::stof(v);
            else if (k == "alignf") alignFactor = std::stof(v);
            else if (k == "tol") relTol = std::stof(v);
            else if (k == "comp") requireSameComposition = b();
            else if (k == "label") requireSameLabel = b();
            else if (k == "species") rejectSpeciesMismatch = b();
            else if (k == "exact") exactDistance = b();
            else if (k == "refine") refineIterations = std::stoi(v);
            else if (k == "max") maxMatches = static_cast<size_t>(std::stoul(v));
            else if (k == "threads") numThreads = std::stoi(v);
            else if (k == "parallel") parallel = b();
            else return false;
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
        using Clock = double;
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
        auto lim = [&](float factor) { return factor * eps * (1.0f + p.relTol) + 1e-6f; };
        const int nt = st.threads;

        // ---- Stage 1: SRV LInf range scan
        Clock t0 = omp_get_wtime();
        std::vector<size_t> ids;
        fm.rangeSearchSrvLinf(q.srv, n, lim(p.srvFactor), queryIdx, nt, ids);
        st.afterSrv = ids.size();
        st.msSrv = (omp_get_wtime() - t0) * 1e3;

        struct Cand
        {
            size_t g;
            float srv, sdv, pdd;
        };
        std::vector<Cand> cands(ids.size());
        for (size_t i = 0; i < ids.size(); ++i)
            cands[i] = {ids[i], 0.f, 0.f, 0.f};

        // ---- Stage 2: Composition, Optional Label, & SDV LInf
        t0 = omp_get_wtime();
        {
            const std::string qComp = p.requireSameComposition ? composition(qAtoms) : std::string();
            const std::string qLabel = p.requireSameLabel ? getLabel(queryIdx) : std::string();
            const float sdvLim = lim(p.sdvFactor);
            std::vector<char> keep(cands.size(), 0);

            auto worker2 = [&](long long i) {
                const MolecularDescriptors *d = fm.descriptor(cands[i].g);
                if (!d)
                    return;
                if (p.requireSameComposition && composition(getAtoms(cands[i].g)) != qComp)
                    return;
                if (p.requireSameLabel && getLabel(cands[i].g) != qLabel)
                    return;
                float s = linf(q.sdv, d->sdv, sdvLim);
                if (s > sdvLim)
                    return;
                cands[i].sdv = s;
                cands[i].srv = linf(q.srv, d->srv, std::numeric_limits<float>::infinity());
                keep[i] = 1;
            };

            if (p.parallel && cands.size() > 4 && nt > 1)
            {
#pragma omp parallel for schedule(static) num_threads(nt)
                for (long long i = 0; i < static_cast<long long>(cands.size()); ++i)
                    worker2(i);
            }
            else
            {
                for (size_t i = 0; i < cands.size(); ++i)
                    worker2(static_cast<long long>(i));
            }
            compact(cands, keep);
        }
        st.afterSdv = cands.size();
        st.msSdv = (omp_get_wtime() - t0) * 1e3;

        if (n > MAXN)
        {
            st.verified = false;
            st.msTotal = (omp_get_wtime() - tAll) * 1e3;
            return out;
        }

        // ---- Stage 3: PDD (Row-wise LInf bottleneck matching via BottleneckEvaluator)
        t0 = omp_get_wtime();
        {
            const float pddLim = lim(p.pddFactor);
            std::vector<char> keep(cands.size(), 0);

            auto worker3 = [&](long long i) {
                const MolecularDescriptors *d = fm.descriptor(cands[i].g);
                float v = 0.f;
                if (d && pddCheck(q.pdd, d->pdd, n, pddLim, v))
                {
                    cands[i].pdd = v;
                    keep[i] = 1;
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

            auto worker4 = [&](long long i) {
                const auto &cAtoms = getAtoms(cands[i].g);
                AlignmentResult res = CloudAlignmentEvaluator::alignAndEvaluate(cloudQ, toCloud(cAtoms), opt);
                if (!res.success || res.bestBottleneck > accept)
                    return;

                int mism = 0;
                for (size_t a = 0; a < res.matching.size() && a < qAtoms.size(); ++a)
                    if (std::string(qAtoms[a].symbol) != std::string(cAtoms[res.matching[a]].symbol))
                        ++mism;
                if (p.rejectSpeciesMismatch && mism > 0)
                    return;

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
        return out;
    }

    // Batch runner for ALL indices in parallel, accumulating evaluation counts and timings
    template <class GetAtoms, class GetLabel = DefaultGetLabel>
    std::vector<DupMatch> runAll(const FaissIndexManager &fm, size_t totalMols,
                                  GetAtoms getAtoms, GetLabel getLabel, const DupCheckParams &p,
                                  DupRunStats &stTotal) const
    {
        std::vector<DupMatch> allMatches;
        std::mutex matchMutex;
        stTotal = DupRunStats{};
        int nt = p.numThreads > 0 ? p.numThreads : omp_get_max_threads();
        stTotal.threads = nt;

        double tStart = omp_get_wtime();
        std::atomic<size_t> processedCount(0);

        size_t accumSrvCount = 0;
        size_t accumSdvCount = 0;
        size_t accumPddCount = 0;
        size_t accumAlignCount = 0;
        double accumMsSrv = 0.0;
        double accumMsSdv = 0.0;
        double accumMsPdd = 0.0;
        double accumMsAlign = 0.0;

#pragma omp parallel for schedule(dynamic, 4) num_threads(nt) reduction(+:accumSrvCount, accumSdvCount, accumPddCount, accumAlignCount, accumMsSrv, accumMsSdv, accumMsPdd, accumMsAlign)
        for (long long i = 0; i < static_cast<long long>(totalMols); ++i)
        {
            const auto *desc = fm.descriptor(i);
            if (!desc || !desc->computed)
                continue;

            const auto &atoms = getAtoms(i);
            DupRunStats localSt;
            std::vector<DupMatch> localMatches = run(fm, i, *desc, atoms, getAtoms, getLabel, p, localSt);

            accumSrvCount += 1;                  // 1 SRV range query per molecule
            accumSdvCount += localSt.afterSrv;   // Candidate pairs entering SDV check
            accumPddCount += localSt.afterSdv;   // Candidate pairs entering PDD check
            accumAlignCount += localSt.afterPdd; // Candidate pairs entering Aligner check

            accumMsSrv += localSt.msSrv;
            accumMsSdv += localSt.msSdv;
            accumMsPdd += localSt.msPdd;
            accumMsAlign += localSt.msAlign;

            if (!localMatches.empty())
            {
                std::lock_guard<std::mutex> lock(matchMutex);
                for (auto &m : localMatches)
                {
                    if (m.queryIdx < m.matchIdx)
                    {
                        allMatches.push_back(std::move(m));
                    }
                }
            }

            size_t done = ++processedCount;
            if (done % 20 == 0 || done == totalMols)
            {
#pragma omp critical
                {
                    std::cout << "\r[DupCheck All] Progress: " << done << "/" << totalMols << " molecules checked..." << std::flush;
                }
            }
        }

        std::cout << "\n";
        stTotal.msTotal = (omp_get_wtime() - tStart) * 1e3;
        stTotal.batchSrvCount = accumSrvCount;
        stTotal.batchSdvCount = accumSdvCount;
        stTotal.batchPddCount = accumPddCount;
        stTotal.batchAlignCount = accumAlignCount;
        stTotal.batchMsSrv = accumMsSrv;
        stTotal.batchMsSdv = accumMsSdv;
        stTotal.batchMsPdd = accumMsPdd;
        stTotal.batchMsAlign = accumMsAlign;

        std::sort(allMatches.begin(), allMatches.end(), [](const DupMatch &a, const DupMatch &b) {
            return a.alignBottleneck < b.alignBottleneck;
        });

        stTotal.afterAlign = allMatches.size();
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
                  << "  1 SRV  Linf <= " << std::setprecision(4) << p.srvFactor * e << "   -> " << s.afterSrv
                  << "   (" << std::setprecision(2) << s.msSrv << " ms)\n"
                  << "  2 SDV  Linf <= " << std::setprecision(4) << p.sdvFactor * e
                  << (p.requireSameComposition ? " +comp" : "")
                  << (p.requireSameLabel ? " +label" : "") << " -> " << s.afterSdv
                  << "   (" << std::setprecision(2) << s.msSdv << " ms)\n";
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
                      << "  " << std::left << std::setw(7) << "match" << std::setw(11) << "align"
                      << std::setw(11) << "pdd" << std::setw(11) << "sdv" << std::setw(11) << "srv"
                      << "species_mismatch\n";
            const size_t lim = std::min<size_t>(m.size(), 15);
            for (size_t i = 0; i < lim; ++i)
                std::cout << "  " << std::left << std::setw(7) << m[i].matchIdx << std::setprecision(6)
                          << std::setw(11) << m[i].alignBottleneck << std::setw(11) << m[i].pddBottleneck
                          << std::setw(11) << m[i].sdvLinf << std::setw(11) << m[i].srvLinf
                          << m[i].speciesMismatches << "\n";
            if (m.size() > lim)
                std::cout << "  ... " << (m.size() - lim) << " more\n";
        }
        std::cout << "=======================================================\n\n";
    }

    // Prints batch summary table with counts and % stage time breakdown for all-indices run
    static void printBatchReport(const DupCheckParams &p, const DupRunStats &s,
                                   const std::vector<DupMatch> &m, const std::string &csvPath)
    {
        double sumStageMs = s.batchMsSrv + s.batchMsSdv + s.batchMsPdd + s.batchMsAlign;
        if (sumStageMs < 1e-3) sumStageMs = 1.0;

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
                  << "    1. SRV Range Searches : " << std::setw(8) << s.batchSrvCount << " queries | " 
                  << std::setw(8) << s.batchMsSrv << " ms (" << std::setprecision(1) << (s.batchMsSrv / sumStageMs) * 100.0 << "%)\n"
                  << "    2. SDV Evaluations    : " << std::setw(8) << s.batchSdvCount << " pairs   | " 
                  << std::setw(8) << s.batchMsSdv << " ms (" << (s.batchMsSdv / sumStageMs) * 100.0 << "%)\n"
                  << "    3. PDD Evaluations    : " << std::setw(8) << s.batchPddCount << " pairs   | " 
                  << std::setw(8) << s.batchMsPdd << " ms (" << (s.batchMsPdd / sumStageMs) * 100.0 << "%)\n"
                  << "    4. Aligned Evaluations: " << std::setw(8) << s.batchAlignCount << " pairs   | " 
                  << std::setw(8) << s.batchMsAlign << " ms (" << (s.batchMsAlign / sumStageMs) * 100.0 << "%)\n"
                  << "-------------------------------------------------------\n";
        if (m.empty())
        {
            std::cout << "  No duplicates found within epsilon threshold.\n";
        }
        else
        {
            std::cout << "  TOP DUPLICATE PAIRS (Sorted by Aligner Distance):\n"
                      << "  " << std::left << std::setw(10) << "MolA"
                      << std::setw(10) << "MolB"
                      << std::setw(12) << "NumAtoms"
                      << std::setw(15) << "AlignerDist"
                      << "SpeciesMismatch\n";
            size_t lim = std::min<size_t>(m.size(), 20);
            for (size_t i = 0; i < lim; ++i)
            {
                std::cout << "  " << std::left << std::setw(10) << m[i].queryIdx
                          << std::setw(10) << m[i].matchIdx
                          << std::setw(12) << m[i].numAtoms
                          << std::setw(15) << std::setprecision(6) << m[i].alignBottleneck
                          << m[i].speciesMismatches << "\n";
            }
            if (m.size() > lim)
                std::cout << "  ... " << (m.size() - lim) << " more pairs\n";
        }
        std::cout << "=======================================================\n\n";
    }

private:
    static float linf(const std::vector<float> &a, const std::vector<float> &b, float limit)
    {
        const size_t n = std::min(a.size(), b.size());
        float m = 0.0f;
        for (size_t k = n; k-- > 0;)
        {
            const float d = std::fabs(a[k] - b[k]);
            if (d > limit)
                return std::numeric_limits<float>::infinity();
            m = std::max(m, d);
        }
        return m;
    }

    template <class Atoms>
    static std::string composition(const Atoms &atoms)
    {
        std::vector<std::string> s;
        s.reserve(atoms.size());
        for (const auto &a : atoms)
            s.push_back(std::string(a.symbol));
        std::sort(s.begin(), s.end());
        std::string r;
        for (const auto &x : s)
        {
            r += x;
            r += ',';
        }
        return r;
    }

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
                v[w++] = v[r];
        v.resize(w);
    }

    static bool pddCheck(const std::vector<float> &pa, const std::vector<float> &pb, size_t n,
                         float thr, float &value)
    {
        if (n == 0 || n > MAXN || pa.size() != n * n || pb.size() != n * n)
            return false;

        float costMatrix[MAXN * MAXN];
        BottleneckEvaluator::computeCostMatrix(pa.data(), pb.data(), n, costMatrix);

        if (!BottleneckEvaluator::hasPerfectMatching(costMatrix, n, thr))
            return false;

        value = BottleneckEvaluator::computeFromCostMatrix(costMatrix, n, 0.0f, thr);
        return true;
    }
};