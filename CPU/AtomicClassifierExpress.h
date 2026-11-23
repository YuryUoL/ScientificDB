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
//    * BATCHED pipeline: Stage-1 candidates are streamed into fixed-size batches
//      (auto-sized from a RAM budget, or fixed manually). Each batch can fire the
//      Stage-2 aligner and free its memory before the next batch is built. This
//      bounds peak RAM to O(batch) + O(survivors), instead of O(all candidates).
//    * Progress tracing + RSS sampling (opt-in, thread-safe, rate-limited).
//      traceProgress = phase markers + count start/end + flush start/end
//      traceVerbose  = per-flush start lines + per-batch detail
//      traceIntervalSec = rate limit for tick/progress lines (default 0.5s)
//    * CSV export of the materialised pair list, with cloud names and atom counts.
//    * Stage-2 bidirectional subset aligner (identical options and defaults to
//      AtomicClassifyParams), so the express result can be verified end-to-end.
//
//  Note on maxMatches:
//    When runAligner == false, maxMatches is ignored (Stage-1 candidates are
//    never truncated). When runAligner == true and topKGlobal == false, the
//    top-K is applied PER BATCH (memory-bounded, non-deterministic order across
//    batches). Set topKGlobal == true to merge survivors and apply a single
//    global top-K at the end (deterministic, RAM = O(all aligned survivors)).
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
#include <mutex>
#include <queue>
#include <omp.h>

#include <unordered_map>

#include "Aligners/CloudAlignmentEvaluator.h"
#include "Aligners/AtomicCloudAlignmentEvaluator.h"
#include "Aligners/AtomicBottleneckEvaluator.h"

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#else
#include <unistd.h>
#endif

#if __has_include(<oneapi/tbb/parallel_sort.h>) && __has_include(<oneapi/tbb/global_control.h>)
#include <oneapi/tbb/parallel_sort.h>
#include <oneapi/tbb/global_control.h>
#define ATOMIC_EXPRESS_HAVE_TBB 1
#endif

// =============================================================================
//  Small utilities: RSS sampling + thread-safe, rate-limited tracer
// =============================================================================
namespace atomic_express_detail
{
    // Resident set size in bytes; 0 if unsupported.
    inline size_t currentRssBytes()
    {
#if defined(_WIN32)
        PROCESS_MEMORY_COUNTERS pmc{};
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
            return static_cast<size_t>(pmc.WorkingSetSize);
        return 0;
#elif defined(__linux__) || defined(__APPLE__)
        std::ifstream f("/proc/self/statm");
        size_t total = 0, rss = 0;
        if (f >> total >> rss)
        {
            const long page = sysconf(_SC_PAGESIZE);
            if (page > 0)
                return rss * static_cast<size_t>(page);
        }
        return 0;
#else
        return 0;
#endif
    }

    // Total physical RAM in bytes; 0 if unknown.
    inline size_t totalRamBytes()
    {
#if defined(_WIN32)
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof(ms);
        if (GlobalMemoryStatusEx(&ms))
            return static_cast<size_t>(ms.ullTotalPhys);
        return 0;
#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
        const long pages = sysconf(_SC_PHYS_PAGES);
        const long page = sysconf(_SC_PAGESIZE);
        if (pages > 0 && page > 0)
            return static_cast<size_t>(pages) * static_cast<size_t>(page);
        return 0;
#else
        return 0;
#endif
    }

    // Thread-safe, rate-limited tracer.
    //
    //  phase(msg)     -> unconditional marker (start/end of a stage)
    //  force(msg)     -> unconditional line (same as phase; kept for clarity)
    //  line(msg)      -> alias of force() — kept for backwards compatibility
    //                    with earlier call sites
    //  tick(msg)      -> prints only if >= minIntervalSec since last print
    //  progress(l,d,t)-> prints "label  x.y%  (d/t)  eta N.Ns"  (rate-limited)
    //
    // NOTE: phase() intentionally takes NO result argument — a previous version
    // forward-declared `struct AtomicExpressResult` inside this namespace, which
    // created a DIFFERENT type from the global ::AtomicExpressResult and broke
    // every call site.  Keeping only the message sidesteps that.
    struct Trace
    {
        bool enabled = false;
        bool mem = false;
        bool verbose = false;
        double minIntervalSec = 0.5;
        double t0 = 0.0;
        mutable double lastPrint = 0.0;
        mutable std::mutex mu;

        Trace() : t0(omp_get_wtime()) {}
        Trace(bool en, bool m) : enabled(en), mem(m), t0(omp_get_wtime()) {}

        void phase(const std::string &msg) const { force(msg); }

        void force(const std::string &msg) const
        {
            if (!enabled)
                return;
            std::lock_guard<std::mutex> lk(mu);
            printLocked(msg);
        }

        // Backwards-compatible alias for force().
        void line(const std::string &msg) const { force(msg); }

        void tick(const std::string &msg) const
        {
            if (!enabled)
                return;
            const double now = omp_get_wtime();
            std::lock_guard<std::mutex> lk(mu);
            if (now - lastPrint < minIntervalSec)
                return;
            lastPrint = now;
            printLocked(msg);
        }

        void progress(const std::string &label, uint64_t done, uint64_t total) const
        {
            if (!enabled || total == 0)
                return;
            const double now = omp_get_wtime();
            std::lock_guard<std::mutex> lk(mu);
            if (now - lastPrint < minIntervalSec)
                return;
            lastPrint = now;
            const double frac = static_cast<double>(done) / static_cast<double>(total);
            const double elapsed = now - t0;
            const double eta = frac > 1e-6 ? elapsed * (1.0 - frac) / frac : 0.0;
            std::ostringstream os;
            os << label << "  " << std::fixed << std::setprecision(1)
               << (frac * 100.0) << "%  (" << done << "/" << total << ")"
               << "  eta " << std::setprecision(1) << eta << "s";
            printLocked(os.str());
        }

    private:
        void printLocked(const std::string &msg) const
        {
            const double t = omp_get_wtime() - t0;
            std::cout << "[trace " << std::fixed << std::setprecision(3) << t << "s] " << msg;
            if (mem)
                std::cout << "  rss=" << (currentRssBytes() >> 20) << "MiB";
            std::cout << std::endl;
        }
    };
} // namespace atomic_express_detail

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

    // ---- Progress / tracing (opt-in) ----
    bool traceProgress = false;
    bool traceMemory = false;
    bool traceVerbose = false;     // per-flush start lines / extra detail
    double traceIntervalSec = 0.5; // rate limit for tick/progress lines
    int traceEveryTables = 1;

    // ---- Batching ----
    // If both batchPairs and batchBytes are 0:
    //   autoTuneRam==true  -> AUTO (sized from free RAM)
    //   autoTuneRam==false -> disabled (legacy: materialize everything)
    //
    // If batchPairs > 0                -> MANUAL, use it
    // If batchBytes > 0 (and pairs==0) -> MANUAL via bytes
    // batchBytes == SIZE_MAX           -> force AUTO path
    size_t batchPairs = 0;
    size_t batchBytes = 0;
    bool pipelineAlign = true; // run aligner per-batch when runAligner == true
    bool topKGlobal = true;    // global top-K (deterministic); false = per-batch
    bool autoTuneRam = true;   // if both batch sizes are 0, auto-pick one

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

    bool storeViz = true;
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

    // ---- Batching telemetry ----
    uint64_t batches = 0;           // number of Stage-2 flushes performed
    uint64_t batchPairsFlushed = 0; // total Stage-1 candidates sent to flushes
    uint64_t peakBatchPairs = 0;    // largest single batch size seen
    size_t peakRssBytes = 0;        // peak RSS observed (only when traceMemory)
    size_t batchBudgetBytes = 0;    // the effective budget used for sizing
    size_t effectiveBatchPairs = 0; // 0 == batching disabled

    // ---- Timing breakdown (all in ms) ----
    double msIndex       = 0;  // stage-0 compaction + grid geometry
    double msSort        = 0;  // per-table entry sort
    double msCells       = 0;  // per-table cell/hash construction
    double msCount       = 0;  // per-table exact SRV counting
    double msMaterialize = 0;  // metadata fill + batch flushes (incl. aligner)
    double msAlign       = 0;  // stage-2 aligner only (subset of msMaterialize)
    double msTotal       = 0;  // wall clock of countCrossLabelPairs() — NO CSV write
    double msWithoutCsv  = 0;  // == msTotal (explicit; writeCsv is a separate call)
    double msCsvWrite    = 0;  // filled only by timedWriteCsv()

    // Materialised pairs (filled only when csvPath is set or runAligner == true).
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
        atomic_express_detail::Trace TR(p.traceProgress, p.traceMemory);
        TR.verbose = p.traceVerbose;
        TR.minIntervalSec = std::max(0.0, p.traceIntervalSec);
        const int nt = p.parallel ? (p.numThreads > 0 ? p.numThreads : omp_get_max_threads()) : 1;
        R.threads = nt;
        R.totalClouds = clouds.size();
#ifdef ATOMIC_EXPRESS_HAVE_TBB
        oneapi::tbb::global_control gc(oneapi::tbb::global_control::max_allowed_parallelism, static_cast<size_t>(nt));
#endif
        const size_t n = clouds.size();
        TR.phase("begin: " + std::to_string(n) + " clouds, threads=" + std::to_string(nt));
        if (n < 2)
            return R;

        // ---- resolve batch sizing ----
        resolveBatchSizing(p, R, TR);

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

        TR.phase("stage0 done (valid clouds compacted): M=" + std::to_string(M) +
                 " T=" + std::to_string(T));

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

        TR.phase("grid geometry ready: G=" + std::to_string(G) +
                 " cellWidth=" + std::to_string(w) +
                 " thr=" + std::to_string(thr));

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

        // ---- timing accumulators for the batch pipeline ----
        double alignAccumMs = 0.0;
        double materializeAccumMs = 0.0;

        // ---- batch pipeline ----
        struct BatchStore
        {
            std::vector<AtomicExpressResult::PairRecord> buf;   // current batch
            std::vector<AtomicExpressResult::PairRecord> all;   // legacy: all
            std::vector<AtomicExpressResult::PairRecord> kept;  // post-align survivors
            uint64_t flushed = 0;
        } B;

        // Fill cloud-level metadata (names, atom counts, indices) for a record.
        auto fillMetadata = [&](AtomicExpressResult::PairRecord &rec)
        {
            const auto &A = clouds[rec.idxA];
            const auto &Bc = clouds[rec.idxB];
            if (p.csvIncludeCloudNames)
            {
                rec.nameA = A.name;
                rec.nameB = Bc.name;
            }
            if (p.csvIncludeAtomCounts)
            {
                rec.atomsInA = A.numAtoms;
                rec.atomsInB = Bc.numAtoms;
            }
            rec.molIdxA = A.parentMoleculeIdx;
            rec.atomIdxA = A.atomIndex;
            rec.molIdxB = Bc.parentMoleculeIdx;
            rec.atomIdxB = Bc.atomIndex;
        };

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
                const float *Rr = out.rotation;
                std::vector<float> cost(mSub * nSub, std::numeric_limits<float>::infinity());
                float lo = 0.0f, hi = 0.0f;
                for (size_t i = 0; i < mSub; ++i)
                {
                    const auto &pA = srcCloud[idxSrc[i]];
                    float ax = Rr[0] * pA.x + Rr[1] * pA.y + Rr[2] * pA.z;
                    float ay = Rr[3] * pA.x + Rr[4] * pA.y + Rr[5] * pA.z;
                    float az = Rr[6] * pA.x + Rr[7] * pA.y + Rr[8] * pA.z;
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

        // Align every record in `v` in place. Only aligned==true rows survive,
        // and their alignAB/alignBA rotations are populated.
        auto alignBatch = [&](std::vector<AtomicExpressResult::PairRecord> &v)
        {
            if (v.empty())
                return;
            const double ta0 = omp_get_wtime();
            std::atomic<uint64_t> alignDone{0};
            const uint64_t alignTotal = static_cast<uint64_t>(v.size());

#pragma omp parallel for schedule(dynamic, 1) num_threads(nt)
            for (long long k = 0; k < static_cast<long long>(v.size()); ++k)
            {
                auto &rec = v[static_cast<size_t>(k)];
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

                    if (maxDist <= rAlign)
                    {
                        rec.aligned = true;
                        rec.alignAB = std::move(rAB);
                        rec.alignBA = std::move(rBA);
                    }
                }

                if (p.traceProgress)
                {
                    const uint64_t c = ++alignDone;
                    if ((c & 0x3FF) == 0) // throttle check every 1024
                        TR.progress("align", c, alignTotal);
                }
            }
            v.erase(std::remove_if(v.begin(), v.end(),
                                   [](const auto &r)
                                   { return !r.aligned; }),
                    v.end());
            if (p.maxMatches > 0 && !p.topKGlobal && v.size() > p.maxMatches)
            {
                std::sort(v.begin(), v.end(),
                          [](const auto &a, const auto &b)
                          { return a.alignDist < b.alignDist; });
                v.resize(p.maxMatches);
            }
            alignAccumMs += (omp_get_wtime() - ta0) * 1e3;
        };

        // Flush the current batch (if any). Metadata is filled here so that
        // even non-aligned survivors carry names/atom counts for CSV/report.
        auto flushBatch = [&](bool finalFlush)
        {
            if (!wantPairs)
            {
                B.buf.clear();
                return;
            }
            const uint64_t inN = B.buf.size();
            if (inN == 0 && !finalFlush)
                return;

            if (p.traceProgress && (p.traceVerbose || finalFlush))
            {
                std::ostringstream os;
                os << (finalFlush ? "final flush" : "flush")
                   << " #" << (R.batches + 1) << " START  N=" << inN;
                TR.force(os.str());
            }

            const double tf0 = omp_get_wtime();
            if (inN)
            {
#pragma omp parallel for schedule(static) num_threads(nt)
                for (long long k = 0; k < static_cast<long long>(B.buf.size()); ++k)
                    fillMetadata(B.buf[static_cast<size_t>(k)]);

                if (p.runAligner && p.pipelineAlign)
                    alignBatch(B.buf);
                // else: raw candidates kept; aligner runs in final pass

                B.kept.insert(B.kept.end(),
                              std::make_move_iterator(B.buf.begin()),
                              std::make_move_iterator(B.buf.end()));
                B.flushed += inN;
                B.buf.clear();
                ++R.batches;
            }
            materializeAccumMs += (omp_get_wtime() - tf0) * 1e3;

            R.batchPairsFlushed = B.flushed;
            R.peakBatchPairs = std::max<uint64_t>(R.peakBatchPairs, inN);
            if (p.traceMemory)
                R.peakRssBytes = std::max<size_t>(R.peakRssBytes,
                                                  atomic_express_detail::currentRssBytes());
            if (p.traceProgress && (p.traceVerbose || finalFlush || inN >= 100000))
            {
                std::ostringstream os;
                os << (finalFlush ? "final flush" : "flush")
                   << " #" << R.batches << " END  in=" << inN
                   << " kept=" << B.kept.size()
                   << " +" << std::fixed << std::setprecision(0)
                   << (omp_get_wtime() - tf0) * 1e3 << "ms";
                TR.force(os.str());
            }
        };

        // Enqueue a batch of records; flush when the batch exceeds the budget.
        auto enqueue = [&](std::vector<AtomicExpressResult::PairRecord> &&incoming)
        {
            if (incoming.empty())
                return;
            if (R.effectiveBatchPairs == 0)
            {
                // legacy: no batching, keep everything
                if (p.runAligner && !p.pipelineAlign)
                {
                    B.all.insert(B.all.end(),
                                 std::make_move_iterator(incoming.begin()),
                                 std::make_move_iterator(incoming.end()));
                }
                else
                {
                    // no batching + no pipeline: materialise directly into kept
                    if (!B.buf.empty())
                    {
#pragma omp parallel for schedule(static) num_threads(nt)
                        for (long long k = 0; k < static_cast<long long>(B.buf.size()); ++k)
                            fillMetadata(B.buf[static_cast<size_t>(k)]);
                        B.kept.insert(B.kept.end(),
                                      std::make_move_iterator(B.buf.begin()),
                                      std::make_move_iterator(B.buf.end()));
                        B.buf.clear();
                    }
                    B.buf.insert(B.buf.end(),
                                 std::make_move_iterator(incoming.begin()),
                                 std::make_move_iterator(incoming.end()));
                }
                return;
            }

            size_t pos = 0;
            while (pos < incoming.size())
            {
                const size_t space = R.effectiveBatchPairs - B.buf.size();
                const size_t take = std::min(space, incoming.size() - pos);
                B.buf.insert(B.buf.end(),
                             std::make_move_iterator(incoming.begin() + pos),
                             std::make_move_iterator(incoming.begin() + pos + take));
                pos += take;
                if (B.buf.size() >= R.effectiveBatchPairs)
                    flushBatch(false);
            }
        };

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
            {
                TR.tick("table g=" + std::to_string(g) + "/" + std::to_string(G) + " skipped (empty)");
                continue;
            }

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

            if (p.traceProgress)
            {
                std::ostringstream os;
                os << "count g=" << g << "/" << G << " START  M=" << nMem << " cells=" << C;
                TR.force(os.str());
            }

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

            std::atomic<uint64_t> countTicks{0};
            const uint64_t totalTicks = static_cast<uint64_t>(big.size() + C);

#pragma omp parallel num_threads(nt)
            {
                Local &L = locals[omp_get_thread_num()];
#pragma omp for schedule(dynamic, 1) nowait
                for (long long t = 0; t < static_cast<long long>(big.size()); ++t)
                {
                    runChunk(big[static_cast<size_t>(t)].cell, big[static_cast<size_t>(t)].e0,
                             big[static_cast<size_t>(t)].e1, L);
                    if (p.traceProgress)
                    {
                        const uint64_t c = ++countTicks;
                        if ((c & 0xFFF) == 0)
                        {
                            std::ostringstream os;
                            os << "count g=" << g << "/" << G;
                            TR.progress(os.str(), c, totalTicks);
                        }
                    }
                }
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

            if (p.traceProgress)
            {
                std::ostringstream os;
                os << "count g=" << g << "/" << G << " END  +"
                   << std::fixed << std::setprecision(0)
                   << (omp_get_wtime() - tCells) * 1e3 << "ms"
                   << "  totalSoFar=" << R.total;
                TR.force(os.str());
            }

            // ---- drain per-thread buffers into the batch pipeline ----
            if (wantPairs)
            {
                std::vector<AtomicExpressResult::PairRecord> incoming;
#pragma omp critical(express_batch_merge)
                {
                    size_t total = 0;
                    for (auto &l : locals)
                        total += l.pairs.size();
                    incoming.reserve(total);
                    for (auto &l : locals)
                    {
                        incoming.insert(incoming.end(),
                                        std::make_move_iterator(l.pairs.begin()),
                                        std::make_move_iterator(l.pairs.end()));
                        l.pairs.clear();
                    }
                }
                enqueue(std::move(incoming));
            }

            if (p.traceProgress && p.traceVerbose)
            {
                std::ostringstream os;
                os << "table g=" << g << "/" << G
                   << " cells=" << C
                   << " maxOcc=" << maxOcc
                   << " flushed=" << B.flushed
                   << " kept=" << B.kept.size();
                TR.line(os.str());
            }
        }

        // ---------------- final flush + assemble ----------------
        if (wantPairs)
        {
            if (B.buf.size() > 0)
                flushBatch(true);

            // Legacy path: no batching but aligner deferred to end -> B.all holds
            // all candidates. Align them once.
            if (R.effectiveBatchPairs == 0 && p.runAligner && !p.pipelineAlign && !B.all.empty())
            {
                TR.force("align (deferred, all candidates) START  N=" + std::to_string(B.all.size()));
#pragma omp parallel for schedule(static) num_threads(nt)
                for (long long k = 0; k < static_cast<long long>(B.all.size()); ++k)
                    fillMetadata(B.all[static_cast<size_t>(k)]);
                alignBatch(B.all);
                B.kept.insert(B.kept.end(),
                              std::make_move_iterator(B.all.begin()),
                              std::make_move_iterator(B.all.end()));
                B.all.clear();
            }

            R.pairs = std::move(B.kept);
            B.kept.clear();

            // final ordering
            if (p.runAligner)
            {
                std::sort(R.pairs.begin(), R.pairs.end(),
                          [](const auto &a, const auto &b)
                          { return a.alignDist < b.alignDist; });
                if (p.topKGlobal && p.maxMatches > 0 && R.pairs.size() > p.maxMatches)
                    R.pairs.resize(p.maxMatches);
            }
            else
            {
                std::sort(R.pairs.begin(), R.pairs.end(),
                          [](const auto &a, const auto &b)
                          { return a.srvDist < b.srvDist; });
            }
        }

        // ---------------- reduce counts ----------------
        R.pairCount.assign(T * T, 0);
        for (const auto &l : locals)
        {
            for (size_t i = 0; i < T * T; ++i)
                R.pairCount[i] += l.cnt[i];
            R.comparisons += l.cmp;
        }
        for (uint64_t v : R.pairCount)
            R.total += v;

        if (p.traceMemory)
            R.peakRssBytes = std::max<size_t>(R.peakRssBytes,
                                              atomic_express_detail::currentRssBytes());

        // ---------------- timing finalize ----------------
        R.msAlign = alignAccumMs;
        R.msMaterialize = wantPairs ? materializeAccumMs : 0.0;
        R.msTotal = (omp_get_wtime() - tAll) * 1e3;
        R.msWithoutCsv = R.msTotal; // CSV write is outside this method

        TR.phase("done  total=" + std::to_string(R.total) +
                 "  in " + std::to_string(R.msTotal) + "ms");
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

    // Wall-clock wrapper around writeCsv() that also records elapsed ms.
    // The write happens OUTSIDE countCrossLabelPairs(); msTotal never includes it.
    static bool timedWriteCsv(const std::string &path, AtomicExpressResult &r)
    {
        const double t0 = omp_get_wtime();
        const bool ok = writeCsv(path, r);
        r.msCsvWrite = (omp_get_wtime() - t0) * 1e3;
        return ok;
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
        if (r.msMaterialize > 0)
            std::cout << " | materialize " << r.msMaterialize;
        if (r.msAlign > 0)
            std::cout << " | align " << r.msAlign;
        std::cout << " | total " << r.msTotal << "\n";

        // Explicit "analysis only, no CSV write" line.
        std::cout << "  Time excluding CSV write: " << r.msWithoutCsv << " ms";
        if (r.msCsvWrite > 0)
            std::cout << "  | CSV write: " << r.msCsvWrite << " ms";
        std::cout << "\n";

        // ---- batching telemetry ----
        if (r.effectiveBatchPairs > 0)
        {
            std::cout << "  Batching: " << r.batches << " flushes, peak "
                      << r.peakBatchPairs << " pairs/flush, budget "
                      << (r.batchBudgetBytes >> 20) << " MiB"
                      << " (batch size " << r.effectiveBatchPairs << ")\n";
            std::cout << "  Total pairs flushed through pipeline: " << r.batchPairsFlushed << "\n";
        }
        else
        {
            std::cout << "  Batching: disabled (all candidates materialised)\n";
        }
        if (r.peakRssBytes > 0)
            std::cout << "  Peak RSS: " << (r.peakRssBytes >> 20) << " MiB\n";
        std::cout << "=======================================================\n";

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
            std::unordered_map<std::string, uint64_t> labelCounts;
            labelCounts.reserve(r.types.size() * r.types.size());

            uint64_t accepted = 0;
            double sumAlign = 0.0;
            float minAlign = std::numeric_limits<float>::infinity();
            float maxAlign = 0.0f;

            for (const auto &pr : r.pairs)
            {
                if (!pr.aligned)
                    continue;
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

private:
    // Resolve batching parameters into a concrete effectiveBatchPairs + budget.
    // Rules (documented in AtomicExpressParams):
    //   * batchPairs > 0                 -> manual, use it
    //   * batchBytes > 0 (and batchPairs==0)
    //       -> batchPairs = batchBytes / ~sizeof(PairRecord)
    //   * both 0 and autoTuneRam == true -> derive from free physical RAM
    //       budget = clamp(free * 0.5, 32 MiB, totalRam/4)
    //       batchPairs = clamp(budget / est, 50_000, 5_000_000)
    //   * both 0 and autoTuneRam == false -> batching disabled (legacy)
    static void resolveBatchSizing(const AtomicExpressParams &p,
                                   AtomicExpressResult &R,
                                   const atomic_express_detail::Trace &TR)
    {
        // Estimated bytes per PairRecord.  Rough: two std::strings (~64 B),
        // two SubsetAlignmentResult (rotation 9 floats + a few size_t + bools,
        // ~80 B each), plus scalar fields -> ~200-260 B.  Use 256 to be safe.
        constexpr size_t kEstPairBytes = 256;

        size_t budget = 0;
        size_t pairs = 0;

        if (p.batchPairs > 0)
        {
            pairs = p.batchPairs;
            budget = pairs * kEstPairBytes;
            TR.line("batch sizing: manual batchPairs=" + std::to_string(pairs));
        }
        else if (p.batchBytes > 0 && p.batchBytes != std::numeric_limits<size_t>::max())
        {
            budget = p.batchBytes;
            pairs = std::max<size_t>(1, budget / kEstPairBytes);
            TR.line("batch sizing: manual batchBytes=" + std::to_string(budget) +
                    " -> " + std::to_string(pairs) + " pairs");
        }
        else if (p.autoTuneRam)
        {
            const size_t total = atomic_express_detail::totalRamBytes();
            const size_t rss = atomic_express_detail::currentRssBytes();
            size_t free_est = 0;
            if (total > rss)
                free_est = total - rss;

            // Default: 50% of free, clamped to [32 MiB, total/4].
            size_t b = free_est ? (free_est / 2) : (size_t(256) << 20);
            const size_t lo = size_t(32) << 20;
            const size_t hi = total ? std::max(lo, total / 4) : (size_t(1024) << 20);
            if (b < lo)
                b = lo;
            if (b > hi)
                b = hi;
            budget = b;
            pairs = std::max<size_t>(1, budget / kEstPairBytes);
            // clamp pair count to sane range
            constexpr size_t kMinPairs = 50'000;
            constexpr size_t kMaxPairs = 5'000'000;
            if (pairs < kMinPairs)
                pairs = kMinPairs;
            if (pairs > kMaxPairs)
                pairs = kMaxPairs;
            budget = pairs * kEstPairBytes;
            TR.line("batch sizing: auto (" + std::to_string(total >> 20) + " MiB total, " +
                    std::to_string(rss >> 20) + " MiB rss) -> budget " +
                    std::to_string(budget >> 20) + " MiB, " + std::to_string(pairs) + " pairs");
        }
        else
        {
            // explicit disable
            pairs = 0;
            budget = 0;
            TR.line("batch sizing: disabled (autoTuneRam=false, no explicit size)");
        }

        R.effectiveBatchPairs = pairs;
        R.batchBudgetBytes = budget;
    }
};