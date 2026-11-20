#pragma once
// =============================================================================
//  AtomicClassifierExpress  -  "light" Stage-1-only SRV pair COUNTER
//
//  Counts (does not materialise by default) all unordered pairs of atomic clouds
//  whose SRV distance
//      D(A,B) = max_{k < min(|A|,|B|)} |A_k - B_k|
//  is within the same threshold the regular AtomicClassifier Stage 1 uses:
//      rSrv = srvFactor * eps * (1 + relTol) + 1e-6      (pair accepted if D <= rSrv)
//  Pairs are broken down by (centre-atom label A, centre-atom label B).
//
//  Semantics preserved from AtomicClassifier::classifyAll:
//    * effective descriptor length  = min(N, sortedDistances.size())   (N==0 -> full)
//    * missing dimensions are unconstrained (only the shared prefix is compared)
//    * considerSame == false -> pairs with identical centre symbol are skipped
//
//  Algorithm (exact, hash grid on a descriptor prefix):
//    G = min(gridDim, longest descriptor), gridDim <= 4 (64-bit key cap).
//    For every depth g = 1..G a pair with s = min(lenA,lenB) is handled in the
//    table of depth min(s, G) -> every pair is visited by exactly one table:
//      * top table (g == G): all clouds with len >= G, half-neighbourhood
//      * lower table (g < G): clouds with len >= g; "E" members (len == g) are the
//        outer loop, "L" members (len > g) are only ever partners of E members
//    Cell key = G coordinates bit-packed into a uint64 (collision free), cells are
//    found by sort + boundary detection, neighbours by an open-addressing hash.
//    Cell width >= threshold, so adjacent cells are sufficient; the final test
//    always uses the original float descriptor values.
//
//  Optional features (all off by default, zero cost when disabled):
//    * CSV export of the materialised pair list, with cloud names and atom counts.
//    * Stage-2 bidirectional subset aligner (identical options and defaults to
//      AtomicClassifyParams), so the express result can be verified end-to-end.
//
//  Note on N > 4:
//    The 64-bit packed key allows at most 4 coordinates at 15 bits each, so the
//    grid dimensionality is hard-capped at 4.  For N > 4 the algorithm remains
//    CORRECT (the exact test still compares all min(lenA,lenB) dimensions) but
//    the grid only partitions the first 4 coordinates -> degradation is purely a
//    matter of cell occupancy, never of counted pairs.
// =============================================================================

#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <memory>
#include <atomic>
#include <limits>
#include <numeric>
#include <array>
#include <cassert>
#include <fstream>
#include <sstream>
#include <omp.h>

#include <unordered_map>

#include "Aligners/CloudAlignmentEvaluator.h"
#include "Aligners/AtomicCloudAlignmentEvaluator.h"
#include "Aligners/AtomicBottleneckEvaluator.h"

#if __has_include(<oneapi/tbb/parallel_sort.h>) && __has_include(<oneapi/tbb/global_control.h>)
#include <oneapi/tbb/parallel_sort.h>
#include <oneapi/tbb/global_control.h>
#define ATOMIC_EXPRESS_HAVE_TBB 1
#endif

// =============================================================================
//  Parameters
// =============================================================================
struct AtomicExpressParams
{
    // ---- Stage-1 core ----
    size_t N = 4;
    float epsilon = 0.10f;
    bool considerSame = false; // false: only pairs with different centre symbols
    float srvFactor = 1.0f;
    float relTol = 1e-4f;
    int numThreads = 0; // 0 = OpenMP default
    bool parallel = true;
    int gridDim = 4; // grid dimensionality cap (1..4)

    // ---- CSV export (all off unless csvPath non-empty) ----
    std::string csvPath; // empty => no CSV
    bool csvIncludeCloudNames = true;
    bool csvIncludeAtomCounts = true;

    // ---- Stage-2 aligner (mirrors AtomicClassifyParams, off by default) ----
    bool runAligner = false;
    bool optimal = false;
    float alignFactor = 1.0f;
    bool requireSameNeighborSpecies = false;
    int refineIterations = 0;
    size_t maxMatches = 0; // 0 = all (only when runAligner == true)
};

// =============================================================================
//  Result
// =============================================================================
struct AtomicExpressResult
{
    std::vector<std::string> types;  // sorted unique centre symbols
    std::vector<uint64_t> pairCount; // T*T matrix, only [a][b] with a<=b used
    uint64_t total = 0;

    size_t totalClouds = 0, validClouds = 0;
    int threads = 1;
    int gridDimUsed = 0;
    float threshold = 0.f;
    double cellWidth = 0.0;

    uint64_t occupiedCells = 0;
    uint64_t comparisons = 0;
    uint64_t maxOccupancy = 0;
    double avgOccupancy = 0.0;

    double msIndex = 0, msSort = 0, msCells = 0, msCount = 0, msAlign = 0, msTotal = 0;

    // Materialised pairs (filled only when csvPath is set or runAligner == true).
    //
    // When runAligner == false:  R.pairs contains every Stage-1 survivor, with
    //                            aligned == false and empty alignAB/alignBA.
    // When runAligner == true:   R.pairs contains only the pairs that survived
    //                            BOTH Stage-1 SRV and Stage-2 bidirectional
    //                            aligner (aligned == true), with valid
    //                            SubsetAlignmentResult rotations in both dirs.
    struct PairRecord
    {
        size_t idxA = 0, idxB = 0;
        std::string nameA, nameB;
        std::string symbolA, symbolB;
        size_t molIdxA = 0, atomIdxA = 0;
        size_t molIdxB = 0, atomIdxB = 0;
        size_t atomsInA = 0, atomsInB = 0;
        size_t lenA = 0, lenB = 0;
        float srvDist = 0.0f;

        bool aligned = false;
        float alignDistAB = 0.0f;
        float alignDistBA = 0.0f;
        float alignDist = 0.0f;
        std::string alignDir = "A->B";

        // Populated only when aligned == true (A->B and B->A respectively).
        SubsetAlignmentResult alignAB;
        SubsetAlignmentResult alignBA;
    };
    std::vector<PairRecord> pairs;

    uint64_t count(size_t a, size_t b) const
    {
        if (a > b)
            std::swap(a, b);
        return pairCount.empty() ? 0 : pairCount[a * types.size() + b];
    }
};

// =============================================================================
//  Classifier
// =============================================================================
class AtomicClassifierExpress
{
public:
    // ------------------------------------------------------------------ main
    template <typename CloudVec>
    AtomicExpressResult countCrossLabelPairs(const CloudVec &clouds, const AtomicExpressParams &p) const
    {
        AtomicExpressResult R;
        const double tAll = omp_get_wtime();
        const int nt = p.parallel ? (p.numThreads > 0 ? p.numThreads : omp_get_max_threads()) : 1;
        R.threads = nt;
        R.totalClouds = clouds.size();
#ifdef ATOMIC_EXPRESS_HAVE_TBB
        oneapi::tbb::global_control gc(oneapi::tbb::global_control::max_allowed_parallelism, static_cast<size_t>(nt));
#endif
        const size_t n = clouds.size();
        if (n < 2)
            return R;

        // ---------------- Stage 0: compact valid clouds into flat arrays ----------------
        double t0 = omp_get_wtime();
        std::vector<uint16_t> lenAll(n, 0);
#pragma omp parallel for schedule(static) num_threads(nt)
        for (long long i = 0; i < static_cast<long long>(n); ++i)
        {
            const auto &c = clouds[static_cast<size_t>(i)];
            if (!c.descriptorsComputed || c.sortedDistances.empty())
                continue;
            size_t sz = c.sortedDistances.size();
            size_t e = (p.N > 0) ? std::min(p.N, sz) : sz;
            lenAll[static_cast<size_t>(i)] = static_cast<uint16_t>(std::min<size_t>(e, 65535));
        }

        std::vector<uint32_t> src;
        src.reserve(n);
        size_t maxLen = 0;
        for (size_t i = 0; i < n; ++i)
            if (lenAll[i])
            {
                src.push_back(static_cast<uint32_t>(i));
                maxLen = std::max<size_t>(maxLen, lenAll[i]);
            }
        const size_t M = src.size();
        R.validClouds = M;
        if (M < 2)
            return R;

        // unique centre symbols
        std::vector<std::string> &uniq = R.types;
#pragma omp parallel num_threads(nt)
        {
            std::vector<std::string> local;
#pragma omp for schedule(static)
            for (long long r = 0; r < static_cast<long long>(M); ++r)
            {
                const std::string &s = clouds[src[static_cast<size_t>(r)]].symbol;
                bool found = false;
                for (const auto &u : local)
                    if (u == s)
                    {
                        found = true;
                        break;
                    }
                if (!found)
                    local.push_back(s);
            }
#pragma omp critical(express_types)
            {
                for (const auto &s : local)
                    if (std::find(uniq.begin(), uniq.end(), s) == uniq.end())
                        uniq.push_back(s);
            }
        }
        std::sort(uniq.begin(), uniq.end());
        const size_t T = uniq.size();
        if (T > 65535)
        {
            std::cerr << "[AtomicExpress] Fatal: >65535 distinct centre symbols, type index overflows uint16_t.\n";
            return R;
        }

        const size_t stride = maxLen;
        std::vector<float> vec(M * stride, 0.0f);
        std::vector<uint16_t> len(M), type(M);
        std::vector<uint32_t> cloudIdOfM(M); // global cloud index for each compact row
#pragma omp parallel for schedule(static) num_threads(nt)
        for (long long r = 0; r < static_cast<long long>(M); ++r)
        {
            const auto &c = clouds[src[static_cast<size_t>(r)]];
            const uint16_t l = lenAll[src[static_cast<size_t>(r)]];
            std::copy(c.sortedDistances.begin(), c.sortedDistances.begin() + l,
                      vec.begin() + static_cast<size_t>(r) * stride);
            len[static_cast<size_t>(r)] = l;
            type[static_cast<size_t>(r)] = static_cast<uint16_t>(
                std::lower_bound(uniq.begin(), uniq.end(), c.symbol) - uniq.begin());
            cloudIdOfM[static_cast<size_t>(r)] = src[static_cast<size_t>(r)];
        }
        std::vector<uint16_t>().swap(lenAll);
        std::vector<uint32_t>().swap(src);

        // ---------------- grid geometry ----------------
        const int G = static_cast<int>(std::min<size_t>(
            static_cast<size_t>(std::max(1, std::min(p.gridDim, 4))), maxLen));
        R.gridDimUsed = G;
        const int bits = 63 / G;
        const uint64_t maxCoord = (bits >= 63) ? ((uint64_t(1) << 62) - 1) : ((uint64_t(1) << bits) - 1);

        float vMin = std::numeric_limits<float>::max(), vMax = std::numeric_limits<float>::lowest();
#pragma omp parallel num_threads(nt)
        {
            float lo = std::numeric_limits<float>::max(), hi = std::numeric_limits<float>::lowest();
#pragma omp for schedule(static)
            for (long long r = 0; r < static_cast<long long>(M); ++r)
            {
                const float *v = vec.data() + static_cast<size_t>(r) * stride;
                const int d = std::min<int>(G, len[static_cast<size_t>(r)]);
                for (int k = 0; k < d; ++k)
                {
                    lo = std::min(lo, v[k]);
                    hi = std::max(hi, v[k]);
                }
            }
#pragma omp critical(express_minmax)
            {
                vMin = std::min(vMin, lo);
                vMax = std::max(vMax, hi);
            }
        }

        const float thr = p.srvFactor * p.epsilon * (1.0f + p.relTol) + 1e-6f;
        R.threshold = thr;
        double w = static_cast<double>(thr) * (1.0 + 1e-5) + 1e-9;
        const double span = static_cast<double>(vMax) - static_cast<double>(vMin);
        if (maxCoord > 16)
        {
            const double denom = static_cast<double>(maxCoord - 8);
            if (denom > 0.0)
                w = std::max(w, span / denom * (1.0 + 1e-9));
        }
        R.cellWidth = w;
        const double invW = 1.0 / w;
        const double vMinD = static_cast<double>(vMin);

        auto coord = [&](float v) -> uint64_t
        {
            return static_cast<uint64_t>(std::floor((static_cast<double>(v) - vMinD) * invW)) + 2u;
        };

        // ---- pair materialisation setup (zero cost if disabled) ----
        const bool wantPairs = !p.csvPath.empty() || p.runAligner;

        struct Local
        {
            std::vector<uint64_t> cnt;
            std::vector<AtomicExpressResult::PairRecord> pairs;
            uint64_t cmp = 0;
        };
        std::vector<Local> locals(nt);
        for (auto &l : locals)
            l.cnt.assign(T * T, 0);

        R.msIndex = (omp_get_wtime() - t0) * 1e3;

        // reusable buffers
        struct Entry
        {
            uint64_t sk;
            uint32_t id;
        };
        std::vector<Entry> entries;
        std::vector<float> sVec;
        std::vector<uint16_t> sLen, sType;
        std::vector<uint32_t> sCloudId;
        std::vector<uint32_t> cellBegin, cellMid;

        const size_t NC = static_cast<size_t>(nt) * 4;

        // ---------------- tables g = 1..G ----------------
        for (int g = 1; g <= G; ++g)
        {
            const bool top = (g == G);

            t0 = omp_get_wtime();
            std::vector<size_t> cntMem(NC, 0), cntE(NC, 0);
#pragma omp parallel for schedule(static) num_threads(nt)
            for (long long c = 0; c < static_cast<long long>(NC); ++c)
            {
                size_t lo = M * static_cast<size_t>(c) / NC, hi = M * (static_cast<size_t>(c) + 1) / NC;
                size_t m = 0, e = 0;
                for (size_t r = lo; r < hi; ++r)
                {
                    if (len[r] >= g)
                    {
                        ++m;
                        if (top || len[r] == g)
                            ++e;
                    }
                }
                cntMem[static_cast<size_t>(c)] = m;
                cntE[static_cast<size_t>(c)] = e;
            }
            size_t nMem = 0, nE = 0;
            std::vector<size_t> off(NC);
            for (size_t c = 0; c < NC; ++c)
            {
                off[c] = nMem;
                nMem += cntMem[c];
                nE += cntE[c];
            }
            if (nE == 0 || nMem < 2)
                continue;

            entries.resize(nMem);
#pragma omp parallel for schedule(static) num_threads(nt)
            for (long long c = 0; c < static_cast<long long>(NC); ++c)
            {
                size_t lo = M * static_cast<size_t>(c) / NC, hi = M * (static_cast<size_t>(c) + 1) / NC;
                size_t o = off[static_cast<size_t>(c)];
                for (size_t r = lo; r < hi; ++r)
                {
                    if (len[r] < g)
                        continue;
                    const float *v = vec.data() + r * stride;
                    uint64_t key = 0;
                    for (int d = 0; d < g; ++d)
                        key |= coord(v[d]) << (bits * d);
                    const uint64_t isL = (!top && len[r] > g) ? 1u : 0u;
                    entries[o++] = {(key << 1) | isL, static_cast<uint32_t>(r)};
                }
            }
            const double tKeys = omp_get_wtime();
            R.msIndex += (tKeys - t0) * 1e3;

            auto cmpEntry = [](const Entry &a, const Entry &b)
            { return a.sk < b.sk; };
#ifdef ATOMIC_EXPRESS_HAVE_TBB
            if (nt > 1)
                oneapi::tbb::parallel_sort(entries.begin(), entries.end(), cmpEntry);
            else
                std::sort(entries.begin(), entries.end(), cmpEntry);
#else
            std::sort(entries.begin(), entries.end(), cmpEntry);
#endif
            const double tSort = omp_get_wtime();
            R.msSort += (tSort - tKeys) * 1e3;

            sVec.resize(nMem * stride);
            sLen.resize(nMem);
            sType.resize(nMem);
            sCloudId.resize(nMem);
#pragma omp parallel for schedule(static) num_threads(nt)
            for (long long q = 0; q < static_cast<long long>(nMem); ++q)
            {
                const size_t r = entries[static_cast<size_t>(q)].id;
                std::copy(vec.begin() + r * stride, vec.begin() + (r + 1) * stride,
                          sVec.begin() + static_cast<size_t>(q) * stride);
                sLen[static_cast<size_t>(q)] = len[r];
                sType[static_cast<size_t>(q)] = type[r];
                sCloudId[static_cast<size_t>(q)] = cloudIdOfM[r];
            }

            // occupied-cell ranges
            std::vector<size_t> cellCnt(NC, 0);
#pragma omp parallel for schedule(static) num_threads(nt)
            for (long long c = 0; c < static_cast<long long>(NC); ++c)
            {
                size_t lo = nMem * static_cast<size_t>(c) / NC, hi = nMem * (static_cast<size_t>(c) + 1) / NC;
                size_t k = 0;
                for (size_t q = lo; q < hi; ++q)
                    if (q == 0 || (entries[q].sk >> 1) != (entries[q - 1].sk >> 1))
                        ++k;
                cellCnt[static_cast<size_t>(c)] = k;
            }
            size_t C = 0;
            std::vector<size_t> cOff(NC);
            for (size_t c = 0; c < NC; ++c)
            {
                cOff[c] = C;
                C += cellCnt[c];
            }
            cellBegin.resize(C + 1);
            cellMid.resize(C);
#pragma omp parallel for schedule(static) num_threads(nt)
            for (long long c = 0; c < static_cast<long long>(NC); ++c)
            {
                size_t lo = nMem * static_cast<size_t>(c) / NC, hi = nMem * (static_cast<size_t>(c) + 1) / NC;
                size_t o = cOff[static_cast<size_t>(c)];
                for (size_t q = lo; q < hi; ++q)
                    if (q == 0 || (entries[q].sk >> 1) != (entries[q - 1].sk >> 1))
                        cellBegin[o++] = static_cast<uint32_t>(q);
            }
            cellBegin[C] = static_cast<uint32_t>(nMem);

            uint64_t maxOcc = 0;
#pragma omp parallel num_threads(nt)
            {
                uint64_t lmax = 0;
#pragma omp for schedule(static)
                for (long long c = 0; c < static_cast<long long>(C); ++c)
                {
                    const size_t b = cellBegin[static_cast<size_t>(c)], e = cellBegin[static_cast<size_t>(c) + 1];
                    auto it = std::partition_point(entries.begin() + b, entries.begin() + e,
                                                   [](const Entry &x)
                                                   { return (x.sk & 1u) == 0u; });
                    cellMid[static_cast<size_t>(c)] = static_cast<uint32_t>(it - entries.begin());
                    lmax = std::max<uint64_t>(lmax, e - b);
                }
#pragma omp critical(express_occ)
                maxOcc = std::max(maxOcc, lmax);
            }

            // hash table
            size_t cap = 16;
            int capBits = 4;
            while (cap < C * 2)
            {
                cap <<= 1;
                ++capBits;
            }
            const size_t mask = cap - 1;
            std::unique_ptr<std::atomic<uint64_t>[]> tabKey(new std::atomic<uint64_t>[cap]());
            std::vector<uint32_t> tabVal(cap);
            auto hashOf = [&](uint64_t k) -> size_t
            { return static_cast<size_t>((k * 0x9E3779B97F4A7C15ull) >> (64 - capBits)); };
#pragma omp parallel for schedule(static) num_threads(nt)
            for (long long c = 0; c < static_cast<long long>(C); ++c)
            {
                const uint64_t key = entries[cellBegin[static_cast<size_t>(c)]].sk >> 1;
                size_t h = hashOf(key);
                for (;;)
                {
                    uint64_t expected = 0;
                    if (tabKey[h].compare_exchange_strong(expected, key, std::memory_order_relaxed))
                    {
                        tabVal[h] = static_cast<uint32_t>(c);
                        break;
                    }
                    h = (h + 1) & mask;
                }
            }
            auto lookup = [&](uint64_t key) -> int64_t
            {
                size_t h = hashOf(key);
                for (;;)
                {
                    const uint64_t k = tabKey[h].load(std::memory_order_relaxed);
                    if (k == key)
                        return static_cast<int64_t>(tabVal[h]);
                    if (k == 0)
                        return -1;
                    h = (h + 1) & mask;
                }
            };

            // neighbour deltas
            std::vector<int64_t> deltas;
            {
                size_t combos = 1;
                for (int d = 0; d < g; ++d)
                    combos *= 3;
                for (size_t m = 0; m < combos; ++m)
                {
                    size_t mm = m;
                    int64_t dl = 0;
                    bool zero = true;
                    for (int d = 0; d < g; ++d)
                    {
                        int o = static_cast<int>(mm % 3) - 1;
                        mm /= 3;
                        if (o != 0)
                            zero = false;
                        dl += static_cast<int64_t>(o) * (static_cast<int64_t>(1) << (bits * d));
                    }
                    if (zero)
                        continue;
                    if (top && dl < 0)
                        continue;
                    deltas.push_back(dl);
                }
            }
            const double tCells = omp_get_wtime();
            R.msCells += (tCells - tSort) * 1e3;

            R.occupiedCells += C;
            R.maxOccupancy = std::max(R.maxOccupancy, maxOcc);
            if (top)
                R.avgOccupancy = static_cast<double>(nMem) / static_cast<double>(C);

            // ---- count phase ----
            const uint16_t *pLen = sLen.data();
            const uint16_t *pType = sType.data();
            const float *pVec = sVec.data();
            const uint32_t *pCloudId = sCloudId.data();
            const bool considerSame = p.considerSame;

            struct Range
            {
                uint32_t b, e;
            };
            static constexpr size_t MAX_NEIGH = 81;

            auto runChunk = [&](size_t c, size_t e0, size_t e1, Local &L)
            {
                const size_t cb = cellBegin[c], ce = cellBegin[c + 1];
                std::array<Range, MAX_NEIGH> nb;
                int nn = 0;
                const uint64_t key = entries[cb].sk >> 1;
                for (int64_t dl : deltas)
                {
                    const int64_t id = lookup(key + static_cast<uint64_t>(dl));
                    if (id < 0)
                        continue;
                    const size_t uid = static_cast<size_t>(id);
                    const uint32_t rb = (top || uid > c) ? cellBegin[uid] : cellMid[uid];
                    const uint32_t re = cellBegin[uid + 1];
                    if (rb < re && nn < static_cast<int>(MAX_NEIGH))
                        nb[static_cast<size_t>(nn++)] = {rb, re};
                }
                uint64_t cmpLocal = 0;
                uint64_t *cnt = L.cnt.data();

                auto scan = [&](uint32_t idA, const float *a, uint16_t la, uint16_t ta, size_t qb, size_t qe)
                {
                    for (size_t q = qb; q < qe; ++q)
                    {
                        const uint16_t tq = pType[q];
                        if (!considerSame && tq == ta)
                            continue;
                        ++cmpLocal;
                        const float *b = pVec + q * stride;
                        const size_t s = std::min<size_t>(la, pLen[q]);
                        bool ok = true;
                        float maxDiff = 0.0f;
                        for (size_t k = 0; k < s; ++k)
                        {
                            const float d = std::fabs(a[k] - b[k]);
                            if (d > thr)
                            {
                                ok = false;
                                break;
                            }
                            if (wantPairs && d > maxDiff)
                                maxDiff = d;
                        }
                        if (!ok)
                            continue;

                        const size_t lo = std::min(ta, tq), hi = std::max(ta, tq);
                        ++cnt[lo * T + hi];

                        if (wantPairs)
                        {
                            AtomicExpressResult::PairRecord rec;

                            // Orient the pair by cloud index FIRST, then carry symbols/lengths
                            // so that symbolA/lenA always describe clouds[rec.idxA] and
                            // symbolB/lenB always describe clouds[rec.idxB].
                            uint32_t idLo = idA;
                            uint32_t idHi = pCloudId[q];
                            uint16_t symLo = ta;
                            uint16_t symHi = tq;
                            uint16_t lenLo = la;
                            uint16_t lenHi = pLen[q];

                            if (idLo > idHi)
                            {
                                std::swap(idLo, idHi);
                                std::swap(symLo, symHi);
                                std::swap(lenLo, lenHi);
                            }

                            rec.idxA = idLo;
                            rec.idxB = idHi;
                            rec.symbolA = uniq[symLo];
                            rec.symbolB = uniq[symHi];
                            rec.lenA = lenLo;
                            rec.lenB = lenHi;
                            rec.srvDist = maxDiff;
                            L.pairs.push_back(std::move(rec));
                        }
                    }
                };

                for (size_t pIdx = cb + e0; pIdx < cb + e1; ++pIdx)
                {
                    const float *a = pVec + pIdx * stride;
                    const uint16_t la = pLen[pIdx], ta = pType[pIdx];
                    const uint32_t idA = pCloudId[pIdx];
                    scan(idA, a, la, ta, pIdx + 1, ce);
                    for (int r = 0; r < nn; ++r)
                        scan(idA, a, la, ta, nb[static_cast<size_t>(r)].b, nb[static_cast<size_t>(r)].e);
                }
                L.cmp += cmpLocal;
            };

            constexpr size_t CHUNK = 128;
            struct Task
            {
                uint32_t cell, e0, e1;
            };
            std::vector<Task> big;
            for (size_t c = 0; c < C; ++c)
            {
                const size_t ne = cellMid[c] - cellBegin[c];
                if (ne > CHUNK)
                    for (size_t s = 0; s < ne; s += CHUNK)
                        big.push_back({static_cast<uint32_t>(c), static_cast<uint32_t>(s),
                                       static_cast<uint32_t>(std::min(ne, s + CHUNK))});
            }

#pragma omp parallel num_threads(nt)
            {
                Local &L = locals[omp_get_thread_num()];
#pragma omp for schedule(dynamic, 1) nowait
                for (long long t = 0; t < static_cast<long long>(big.size()); ++t)
                    runChunk(big[static_cast<size_t>(t)].cell, big[static_cast<size_t>(t)].e0,
                             big[static_cast<size_t>(t)].e1, L);
#pragma omp for schedule(dynamic, 64)
                for (long long c = 0; c < static_cast<long long>(C); ++c)
                {
                    const size_t ne = cellMid[static_cast<size_t>(c)] - cellBegin[static_cast<size_t>(c)];
                    if (ne == 0 || ne > CHUNK)
                        continue;
                    runChunk(static_cast<size_t>(c), 0, ne, L);
                }
            }
            R.msCount += (omp_get_wtime() - tCells) * 1e3;
        }

        // ---------------- reduce ----------------
        R.pairCount.assign(T * T, 0);
        for (const auto &l : locals)
        {
            for (size_t i = 0; i < T * T; ++i)
                R.pairCount[i] += l.cnt[i];
            R.comparisons += l.cmp;
        }
        for (uint64_t v : R.pairCount)
            R.total += v;

        // ---------------- materialise pairs (metadata + optional aligner) ----------------
        if (wantPairs)
        {
            size_t totalPairs = 0;
            for (auto &l : locals)
                totalPairs += l.pairs.size();
            R.pairs.reserve(totalPairs);
            for (auto &l : locals)
            {
                R.pairs.insert(R.pairs.end(),
                               std::make_move_iterator(l.pairs.begin()),
                               std::make_move_iterator(l.pairs.end()));
                std::vector<AtomicExpressResult::PairRecord>().swap(l.pairs);
            }

            // fill per-cloud metadata (names / counts / indices)
#pragma omp parallel for schedule(static) num_threads(nt)
            for (long long k = 0; k < static_cast<long long>(R.pairs.size()); ++k)
            {
                auto &rec = R.pairs[static_cast<size_t>(k)];
                const auto &A = clouds[rec.idxA];
                const auto &B = clouds[rec.idxB];
                if (p.csvIncludeCloudNames)
                {
                    rec.nameA = A.name;
                    rec.nameB = B.name;
                }
                if (p.csvIncludeAtomCounts)
                {
                    rec.atomsInA = A.numAtoms;
                    rec.atomsInB = B.numAtoms;
                }
                rec.molIdxA = A.parentMoleculeIdx;
                rec.atomIdxA = A.atomIndex;
                rec.molIdxB = B.parentMoleculeIdx;
                rec.atomIdxB = B.atomIndex;
            }

            std::sort(R.pairs.begin(), R.pairs.end(),
                      [](const auto &a, const auto &b)
                      { return a.srvDist < b.srvDist; });

            // ---- optional Stage-2 bidirectional aligner ----
            if (p.runAligner && !R.pairs.empty())
            {
                const double tAlign0 = omp_get_wtime();
                const float rAlign = p.alignFactor * p.epsilon * (1.0f + p.relTol) + 1e-6f;

                const auto MAXN = AtomicBottleneckEvaluator::MAX_VERTICES;

                auto evalDirected = [&](const auto &srcAc,
                                        const std::vector<CloudAlignmentEvaluator::Point3F> &srcCloud,
                                        const std::vector<std::string> &srcSyms,
                                        const auto &dstAc,
                                        const std::vector<CloudAlignmentEvaluator::Point3F> &dstCloud,
                                        const std::vector<std::string> &dstSyms,
                                        SubsetAlignmentResult &out,
                                        float &outDist) -> bool
                {
                    size_t limitSrc = (p.N > 0) ? std::min(p.N, srcAc.sortedAtomIndices.size())
                                                : srcAc.sortedAtomIndices.size();
                    const auto &idxDst = dstAc.sortedAtomIndices;
                    if (limitSrc == 0 || idxDst.empty())
                        return false;
                    if (idxDst.size() < limitSrc)
                        limitSrc = idxDst.size();
                    std::vector<size_t> idxSrc(srcAc.sortedAtomIndices.begin(),
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

                    out = AtomicCloudAlignmentEvaluator::alignSubsetAndEvaluate(
                        srcCloud, idxSrc, dstCloud, idxDst, opt);
                    if (!out.success)
                        return false;
                    outDist = out.bestBottleneck;

                    const size_t mSub = idxSrc.size();
                    const size_t nSub = idxDst.size();
                    if (p.optimal && mSub <= MAXN && nSub <= MAXN && outDist <= rAlign)
                    {
                        // exact injective bottleneck refinement
                        const float *R = out.rotation;
                        std::vector<float> cost(mSub * nSub, std::numeric_limits<float>::infinity());
                        float lo = 0.0f, hi = 0.0f;
                        for (size_t i = 0; i < mSub; ++i)
                        {
                            const auto &pA = srcCloud[idxSrc[i]];
                            float ax = R[0] * pA.x + R[1] * pA.y + R[2] * pA.z;
                            float ay = R[3] * pA.x + R[4] * pA.y + R[5] * pA.z;
                            float az = R[6] * pA.x + R[7] * pA.y + R[8] * pA.z;
                            float rowMin = std::numeric_limits<float>::infinity();
                            float rowMax = 0.0f;
                            for (size_t j = 0; j < nSub; ++j)
                            {
                                if (p.requireSameNeighborSpecies &&
                                    srcSyms[idxSrc[i]] != dstSyms[idxDst[j]])
                                    continue;
                                const auto &pB = dstCloud[idxDst[j]];
                                float dx = ax - pB.x, dy = ay - pB.y, dz = az - pB.z;
                                float d = std::sqrt(dx * dx + dy * dy + dz * dz);
                                cost[i * nSub + j] = d;
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
                            outDist = AtomicBottleneckEvaluator::bottleneckInjective(
                                cost.data(), mSub, nSub, lo, hi);
                        out.bestBottleneck = outDist;
                    }
                    return true;
                };

#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
                for (long long k = 0; k < static_cast<long long>(R.pairs.size()); ++k)
                {
                    auto &rec = R.pairs[static_cast<size_t>(k)];
                    const auto &acA = clouds[rec.idxA];
                    const auto &acB = clouds[rec.idxB];

                    std::vector<CloudAlignmentEvaluator::Point3F> cloudA(acA.atoms.size()), cloudB(acB.atoms.size());
                    std::vector<std::string> symsA, symsB;
                    if (p.requireSameNeighborSpecies)
                    {
                        symsA.resize(acA.atoms.size());
                        symsB.resize(acB.atoms.size());
                    }
                    for (size_t i = 0; i < acA.atoms.size(); ++i)
                    {
                        cloudA[i] = {(float)acA.atoms[i].x, (float)acA.atoms[i].y, (float)acA.atoms[i].z};
                        if (p.requireSameNeighborSpecies)
                            symsA[i] = acA.atoms[i].symbol;
                    }
                    for (size_t i = 0; i < acB.atoms.size(); ++i)
                    {
                        cloudB[i] = {(float)acB.atoms[i].x, (float)acB.atoms[i].y, (float)acB.atoms[i].z};
                        if (p.requireSameNeighborSpecies)
                            symsB[i] = acB.atoms[i].symbol;
                    }

                    SubsetAlignmentResult rAB, rBA;
                    float dAB = std::numeric_limits<float>::infinity();
                    float dBA = std::numeric_limits<float>::infinity();

                    bool okAB = evalDirected(acA, cloudA, symsA, acB, cloudB, symsB, rAB, dAB);
                    bool okBA = evalDirected(acB, cloudB, symsB, acA, cloudA, symsA, rBA, dBA);

                    if (okAB && okBA)
                    {
                        const float maxDist = std::max(dAB, dBA);
                        rec.alignDistAB = dAB;
                        rec.alignDistBA = dBA;
                        rec.alignDist = maxDist;
                        if (dAB > dBA)
                            rec.alignDir = "A->B";
                        else if (dBA > dAB)
                            rec.alignDir = "B->A";
                        else
                            rec.alignDir = "equal";

                        // Mark as fully aligned (and persist the rotations) ONLY when
                        // the bidirectional bottleneck is within the aligner threshold.
                        // Unaligned pairs still carry distances for diagnostics, but
                        // their alignAB/alignBA stay empty so downstream code (viz,
                        // apply_atomic_align, etc.) skips them.
                        if (maxDist <= rAlign)
                        {
                            rec.aligned = true;
                            rec.alignAB = std::move(rAB);
                            rec.alignBA = std::move(rBA);
                        }
                    }
                }

                R.msAlign = (omp_get_wtime() - tAlign0) * 1e3;

                // ---- Always drop unaligned rows after the aligner ran (matches
                //      AtomicClassifier::classifyAll semantics), then optionally
                //      truncate to the top-K best aligned pairs. ----
                R.pairs.erase(
                    std::remove_if(R.pairs.begin(), R.pairs.end(),
                                   [](const auto &r)
                                   { return !r.aligned; }),
                    R.pairs.end());
                std::sort(R.pairs.begin(), R.pairs.end(),
                          [](const auto &a, const auto &b)
                          { return a.alignDist < b.alignDist; });
                if (p.maxMatches > 0 && R.pairs.size() > p.maxMatches)
                    R.pairs.resize(p.maxMatches);
            }
        }

        R.msTotal = (omp_get_wtime() - tAll) * 1e3;
        return R;
    }

    // ----------------------------------------------- O(M^2) correctness oracle
    template <typename CloudVec>
    static AtomicExpressResult countBruteForce(const CloudVec &clouds, const AtomicExpressParams &p)
    {
        AtomicExpressResult R;
        const float thr = p.srvFactor * p.epsilon * (1.0f + p.relTol) + 1e-6f;
        std::vector<size_t> ids;
        for (size_t i = 0; i < clouds.size(); ++i)
            if (clouds[i].descriptorsComputed && !clouds[i].sortedDistances.empty())
            {
                ids.push_back(i);
                R.types.push_back(clouds[i].symbol);
            }
        std::sort(R.types.begin(), R.types.end());
        R.types.erase(std::unique(R.types.begin(), R.types.end()), R.types.end());
        const size_t T = R.types.size();
        R.pairCount.assign(T * T, 0);
        auto tid = [&](const std::string &s)
        { return static_cast<size_t>(std::lower_bound(R.types.begin(), R.types.end(), s) - R.types.begin()); };
        for (size_t a = 0; a < ids.size(); ++a)
            for (size_t b = a + 1; b < ids.size(); ++b)
            {
                const auto &A = clouds[ids[a]], &B = clouds[ids[b]];
                if (!p.considerSame && A.symbol == B.symbol)
                    continue;
                size_t la = (p.N > 0) ? std::min(p.N, A.sortedDistances.size()) : A.sortedDistances.size();
                size_t lb = (p.N > 0) ? std::min(p.N, B.sortedDistances.size()) : B.sortedDistances.size();
                size_t s = std::min(la, lb);
                bool ok = true;
                for (size_t k = 0; k < s && ok; ++k)
                    if (std::fabs(A.sortedDistances[k] - B.sortedDistances[k]) > thr)
                        ok = false;
                if (!ok)
                    continue;
                size_t x = tid(A.symbol), y = tid(B.symbol);
                if (x > y)
                    std::swap(x, y);
                ++R.pairCount[x * T + y];
                ++R.total;
            }
        R.validClouds = ids.size();
        return R;
    }

    // ----------------------------------------------------------------- CSV
    static bool writeCsv(const std::string &path, const AtomicExpressResult &r)
    {
        if (path.empty())
            return false;
        std::ofstream csv(path);
        if (!csv.is_open())
        {
            std::cerr << "[AtomicExpress] Could not open CSV for writing: " << path << "\n";
            return false;
        }
        csv << "CloudIdxA,CloudNameA,MolIdxA,AtomIdxA,SymbolA,AtomsA,LenA,"
            << "CloudIdxB,CloudNameB,MolIdxB,AtomIdxB,SymbolB,AtomsB,LenB,"
            << "SRVDist,Aligned,AlignDistAB,AlignDistBA,AlignDist,AlignDir\n";

        for (const auto &p : r.pairs)
        {
            csv << p.idxA << ',' << p.nameA << ',' << p.molIdxA << ',' << p.atomIdxA << ','
                << p.symbolA << ',' << p.atomsInA << ',' << p.lenA << ','
                << p.idxB << ',' << p.nameB << ',' << p.molIdxB << ',' << p.atomIdxB << ','
                << p.symbolB << ',' << p.atomsInB << ',' << p.lenB << ','
                << std::fixed << std::setprecision(6) << p.srvDist << ','
                << (p.aligned ? 1 : 0) << ','
                << p.alignDistAB << ',' << p.alignDistBA << ',' << p.alignDist << ','
                << p.alignDir << '\n';
        }
        std::cout << "[AtomicExpress] Wrote " << r.pairs.size()
                  << " pairs to " << path << "\n";
        return true;
    }

    // ----------------------------------------------------------------- report
    static void printReport(const AtomicExpressParams &p, const AtomicExpressResult &r)
    {
        std::cout << "\n=======================================================\n"
                  << "  ATOMIC EXPRESS SRV DUPLICATE COUNT (Stage 1 only)\n"
                  << "=======================================================\n"
                  << "  N=" << p.N << "  eps=" << p.epsilon
                  << "  considerSame=" << (p.considerSame ? "true" : "false")
                  << "  threads=" << r.threads << "\n"
                  << "  Valid clouds: " << r.validClouds << " / " << r.totalClouds << "\n"
                  << "-------------------------------------------------------\n"
                  << "  Overall duplicates: " << r.total << "\n";

        struct Row
        {
            std::string name;
            uint64_t n;
        };
        std::vector<Row> rows;
        const size_t T = r.types.size();
        for (size_t a = 0; a < T; ++a)
            for (size_t b = a; b < T; ++b)
                if (r.pairCount[a * T + b])
                    rows.push_back({r.types[a] + "-" + r.types[b], r.pairCount[a * T + b]});
        std::sort(rows.begin(), rows.end(), [](const Row &x, const Row &y)
                  { return x.n > y.n; });
        for (const auto &row : rows)
            std::cout << "    " << std::left << std::setw(8) << row.name << ": " << row.n << "\n";

        std::cout << "-------------------------------------------------------\n"
                  << std::fixed << std::setprecision(2)
                  << "  Grid dim " << r.gridDimUsed << ", cell width " << std::setprecision(5) << r.cellWidth
                  << ", threshold " << r.threshold << std::setprecision(2) << "\n"
                  << "  Occupied cells: " << r.occupiedCells << " | max occupancy: " << r.maxOccupancy
                  << " | avg occupancy (top table): " << r.avgOccupancy << "\n"
                  << "  Candidate comparisons: " << r.comparisons << "\n"
                  << "  Time ms: index " << r.msIndex << " | sort " << r.msSort << " | cells+hash " << r.msCells
                  << " | count " << r.msCount;
        if (r.msAlign > 0)
            std::cout << " | align " << r.msAlign;
        std::cout << " | total " << r.msTotal << "\n"
                  << "=======================================================\n";

        if (!r.pairs.empty())
        {
            std::cout << "  Materialised pairs: " << r.pairs.size() << "\n";
            const bool hasAlign = r.pairs.front().aligned;
            std::cout << "  TOP CLOSE ATOMIC NEIGHBORHOODS (by "
                      << (hasAlign ? "alignDist" : "srvDist") << "):\n"
                      << "  " << std::left << std::setw(22) << "CloudA"
                      << std::setw(7) << "SymA"
                      << std::setw(22) << "CloudB"
                      << std::setw(7) << "SymB"
                      << std::setw(10) << "SRV"
                      << std::setw(12) << "Align" << "Dir\n";
            const size_t lim = std::min<size_t>(r.pairs.size(), 20);
            for (size_t i = 0; i < lim; ++i)
            {
                const auto &pr = r.pairs[i];
                std::cout << "  " << std::left << std::setw(22) << pr.nameA
                          << std::setw(7) << pr.symbolA
                          << std::setw(22) << pr.nameB
                          << std::setw(7) << pr.symbolB
                          << std::fixed << std::setprecision(6)
                          << std::setw(10) << pr.srvDist
                          << std::setw(12) << (pr.aligned ? pr.alignDist : -1.0f)
                          << pr.alignDir << "\n";
            }
            if (r.pairs.size() > lim)
                std::cout << "  ... " << (r.pairs.size() - lim) << " more pairs\n";
        }
        if (p.runAligner && !r.pairs.empty())
        {
            // Tally accepted pairs by (symbolA, symbolB), reusing the same
            // ordering convention as Stage-1: min label first.
            std::unordered_map<std::string, uint64_t> labelCounts;
            labelCounts.reserve(r.types.size() * r.types.size());

            uint64_t accepted = 0;
            double sumAlign = 0.0;
            float minAlign = std::numeric_limits<float>::infinity();
            float maxAlign = 0.0f;

            for (const auto &pr : r.pairs)
            {
                if (!pr.aligned)
                    continue; // defensive: post-filter R.pairs should already be aligned-only
                ++accepted;

                const std::string &a = pr.symbolA;
                const std::string &b = pr.symbolB;
                const std::string key = (a <= b) ? (a + "-" + b) : (b + "-" + a);
                ++labelCounts[key];

                sumAlign += pr.alignDist;
                minAlign = std::min(minAlign, pr.alignDist);
                maxAlign = std::max(maxAlign, pr.alignDist);
            }

            if (accepted > 0)
            {
                std::vector<std::pair<std::string, uint64_t>> rows2(labelCounts.begin(), labelCounts.end());
                std::sort(rows2.begin(), rows2.end(),
                          [](const auto &x, const auto &y)
                          { return x.second > y.second; });

                const double meanAlign = sumAlign / static_cast<double>(accepted);

                std::cout << "-------------------------------------------------------\n"
                          << "  STAGE-2 ALIGNER SURVIVORS (bidirectional bottleneck <= "
                          << std::fixed << std::setprecision(6)
                          << (p.alignFactor * p.epsilon * (1.0f + p.relTol) + 1e-6f)
                          << ")\n"
                          << "-------------------------------------------------------\n"
                          << "  Aligned pairs       : " << accepted
                          << "  (Stage-1 survivors: " << r.total << ")\n"
                          << "  Align dist stats    : min=" << minAlign
                          << "  mean=" << meanAlign
                          << "  max=" << maxAlign << "\n"
                          << "-------------------------------------------------------\n";
                for (const auto &row : rows2)
                    std::cout << "    " << std::left << std::setw(8) << row.first
                              << ": " << row.second << "\n";
            }
        }
        std::cout << "=======================================================\n\n";
    }
};