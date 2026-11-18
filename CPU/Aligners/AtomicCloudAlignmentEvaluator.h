#pragma once
#include "CloudAlignmentEvaluator.h"    // reuses Point3F, AlignmentOptions, frame helpers
#include "AtomicBottleneckEvaluator.h"
#include "BottleneckEvaluator.h"

struct SubsetAlignmentResult
{
    float bestBottleneck = std::numeric_limits<float>::max();
    size_t bestB1 = 0;                     // index into the FULL cloud B
    size_t bestB2 = 0;
    float rotation[9] = {0};               // row-major, p_B ~= R * p_A (no translation)
    std::vector<float> bestRotatedPointsA; // xyz interleaved, for selected A, in idxA order
    std::vector<size_t> matching;          // matching[i] = index in FULL cloud B matched to cloudA[idxA[i]]
    bool success = false;
    int refineIterationsUsed = 0;
};

class AtomicCloudAlignmentEvaluator {
public:
    using Point3F = CloudAlignmentEvaluator::Point3F;

    static SubsetAlignmentResult alignSubsetAndEvaluate(
        const std::vector<Point3F> &cloudA, const std::vector<size_t> &idxA,
        const std::vector<Point3F> &cloudB, const std::vector<size_t> &idxB,
        const AlignmentOptions &opt = AlignmentOptions())
    {
        constexpr size_t MAXV = BottleneckEvaluator::MAX_VERTICES;
        SubsetAlignmentResult result;
        const size_t m = idxA.size(), n = idxB.size();
        if (m < 2 || m > n || n > MAXV)
            return result;
        for (size_t i : idxA)
            if (i >= cloudA.size())
                return result;
        for (size_t j : idxB)
            if (j >= cloudB.size())
                return result;

        // ---------- 1. Select points (raw, uncentered) ----------
        std::vector<Point3F> A(m), B(n);
        for (size_t i = 0; i < m; ++i)
            A[i] = cloudA[idxA[i]];
        for (size_t j = 0; j < n; ++j)
            B[j] = cloudB[idxB[j]];

        const bool checkSymbols = opt.requireSameSymbol && opt.symbolsA && opt.symbolsB;
        auto symOK = [&](size_t i, size_t j) -> bool
        { return (*opt.symbolsA)[idxA[i]] == (*opt.symbolsB)[idxB[j]]; };

        // ---------- 2. Reference triangle in A ----------
        size_t a1, a2;
        if (!CloudAlignmentEvaluator::findBestTriangleA(A, a1, a2))
            return result;
        float Fa[9];
        if (!CloudAlignmentEvaluator::makeFrame(A[a1], A[a2], Fa))
            return result;

        // ---------- 3. Precomputation ----------
        std::array<float, MAXV> qx, qy, qz, bx, by, bz, na2, nb2;
        for (size_t i = 0; i < m; ++i)
        {
            const auto &p = A[i];
            qx[i] = Fa[0] * p.x + Fa[3] * p.y + Fa[6] * p.z;
            qy[i] = Fa[1] * p.x + Fa[4] * p.y + Fa[7] * p.z;
            qz[i] = Fa[2] * p.x + Fa[5] * p.y + Fa[8] * p.z;
            na2[i] = p.x * p.x + p.y * p.y + p.z * p.z;
        }
        for (size_t j = 0; j < n; ++j)
        {
            bx[j] = B[j].x;
            by[j] = B[j].y;
            bz[j] = B[j].z;
            nb2[j] = B[j].x * B[j].x + B[j].y * B[j].y + B[j].z * B[j].z;
        }

        // Lower bound: sorted norms
        float lbGlobal = 0.0f, scale = 0.0f;
        {
            std::vector<float> sa(m), sb(n);
            for (size_t i = 0; i < m; ++i)
                sa[i] = std::sqrt(na2[i]);
            for (size_t j = 0; j < n; ++j)
                sb[j] = std::sqrt(nb2[j]);
            std::sort(sa.begin(), sa.end());
            std::sort(sb.begin(), sb.end());
            for (size_t i = 0; i < m; ++i)
                lbGlobal = std::max(lbGlobal, std::abs(sa[i] - sb[i]));
            scale = std::max(sa[m - 1], sb[n - 1]);
        }
        const float lbSearchSq = std::pow(std::max(0.0f, lbGlobal - 1e-5f * scale), 2.0f);
        float stopSq = std::numeric_limits<float>::max();
        if (opt.acceptDistance > 0.0f)
            stopSq = opt.acceptDistance * opt.acceptDistance;
        else if (opt.exact)
            stopSq = lbGlobal * lbGlobal;

        std::array<uint8_t, MAXV> order;
        for (size_t i = 0; i < m; ++i)
            order[i] = static_cast<uint8_t>(i);
        std::sort(order.begin(), order.begin() + m, [&](uint8_t x, uint8_t y)
                  { return na2[x] > na2[y]; });

        // ---------- 4. Rank ordered candidate pairs (b1,b2) from selected B ----------
        const float s1 = na2[a1], s2 = na2[a2];
        const float dotA = A[a1].x * A[a2].x + A[a1].y * A[a2].y + A[a1].z * A[a2].z;
        const float d12A = s1 + s2 - 2.0f * dotA;
        const float areaA = std::sqrt(std::max(0.0f, s1 * s2 - dotA * dotA));

        struct Cand
        {
            float cost;
            uint16_t b1, b2;
        };
        std::vector<Cand> cands;
        cands.reserve(n * (n - 1));

        for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = i + 1; j < n; ++j)
            {
                const float dot = bx[i] * bx[j] + by[i] * by[j] + bz[i] * bz[j];
                const float crossSq = nb2[i] * nb2[j] - dot * dot;
                if (crossSq <= 1e-10f * nb2[i] * nb2[j])
                    continue;
                const float d12B = nb2[i] + nb2[j] - 2.0f * dot;
                const float areaB = std::sqrt(crossSq);
                const float common = std::abs(d12A - d12B) + std::abs(areaA - areaB);

                if (!checkSymbols || (symOK(a1, i) && symOK(a2, j)))
                    cands.push_back({common + std::abs(s1 - nb2[i]) + std::abs(s2 - nb2[j]),
                                     static_cast<uint16_t>(i), static_cast<uint16_t>(j)});
                if (!checkSymbols || (symOK(a2, i) && symOK(a1, j)))
                    cands.push_back({common + std::abs(s1 - nb2[j]) + std::abs(s2 - nb2[i]),
                                     static_cast<uint16_t>(j), static_cast<uint16_t>(i)});
            }
        }
        if (cands.empty())
            return result;

        auto byCost = [](const Cand &x, const Cand &y)
        { return x.cost < y.cost; };
        if (opt.maxCandidates > 0 && opt.maxCandidates < cands.size())
        {
            std::partial_sort(cands.begin(), cands.begin() + opt.maxCandidates, cands.end(), byCost);
            cands.resize(opt.maxCandidates);
        }
        else
            std::sort(cands.begin(), cands.end(), byCost);

        // ---------- 5. Evaluate candidates ----------
        float cost[MAXV * MAXV];
        uint64_t adj[MAXV];

        auto buildRows = [&](const float Fb[9], float t, float &rowMinMax) -> bool
        {
            rowMinMax = 0.0f;
            for (size_t k = 0; k < m; ++k)
            {
                const size_t i = order[k];
                const float px = Fb[0] * qx[i] + Fb[1] * qy[i] + Fb[2] * qz[i];
                const float py = Fb[3] * qx[i] + Fb[4] * qy[i] + Fb[5] * qz[i];
                const float pz = Fb[6] * qx[i] + Fb[7] * qy[i] + Fb[8] * qz[i];
                float *row = &cost[i * n];
                float mn = std::numeric_limits<float>::max();
                for (size_t j = 0; j < n; ++j)
                {
                    if (checkSymbols && !symOK(i, j))
                    {
                        row[j] = std::numeric_limits<float>::max();
                        continue;
                    }
                    const float dx = px - bx[j], dy = py - by[j], dz = pz - bz[j];
                    const float d = dx * dx + dy * dy + dz * dz;
                    row[j] = d;
                    mn = std::min(mn, d);
                }
                if (mn > t)
                    return false;
                rowMinMax = std::max(rowMinMax, mn);
                uint64_t mask = 0;
                for (size_t j = 0; j < n; ++j)
                    mask |= (uint64_t)(row[j] <= t) << j;
                adj[i] = mask;
            }
            return true;
        };

        float bestSq = std::numeric_limits<float>::max();
        size_t bestC = 0;
        bool found = false;

        for (size_t c = 0; c < cands.size(); ++c)
        {
            float Fb[9];
            if (!CloudAlignmentEvaluator::makeFrame(B[cands[c].b1], B[cands[c].b2], Fb))
                continue;

            const float t = found ? std::nextafter(bestSq, 0.0f) : std::numeric_limits<float>::max();
            float rowMinMax;
            if (!buildRows(Fb, t, rowMinMax))
                continue;
            if (!CloudAlignmentEvaluator::matchFromMasks(adj, m, nullptr))
                continue;

            const float lb = std::max(rowMinMax, lbSearchSq);
            const float v = BottleneckEvaluator::computeFromCostMatrix(cost, m, std::min(lb, t), t);
            if (v < bestSq)
            {
                bestSq = v;
                bestC = c;
                found = true;
                if (!opt.exact && opt.acceptDistance == 0.0f)
                    break;
                if (bestSq <= stopSq)
                    break;
            }
        }
        if (!found)
            return result;

        // ---------- 6. Materialize winner & refine ----------
        const size_t b1 = cands[bestC].b1, b2 = cands[bestC].b2;
        float M[9];
        if (!CloudAlignmentEvaluator::makeFrame(B[b1], B[b2], M))
            return result;

        auto extractMatching = [&](const float Mx[9], float v, std::vector<size_t> &out) -> bool
        {
            float rm;
            float t = v;
            if (!buildRows(Mx, t, rm))
            {
                t = v * (1.0f + 1e-5f) + 1e-30f;
                if (!buildRows(Mx, t, rm))
                    return false;
            }
            uint8_t matchR[MAXV];
            if (!CloudAlignmentEvaluator::matchFromMasks(adj, m, matchR))
                return false;
            out.assign(m, 0);
            for (size_t col = 0; col < n; ++col)
                if (matchR[col] != 255)
                    out[matchR[col]] = col;
            return true;
        };

        std::vector<size_t> matching;
        const bool haveMatching = extractMatching(M, bestSq, matching);

        int iters = 0;
        if (opt.refineIterations > 0 && haveMatching && bestSq > stopSq)
        {
            for (int it = 0; it < opt.refineIterations; ++it)
            {
                float Rk[9], Mn[9];
                if (!CloudAlignmentEvaluator::kabsch(A, B, matching, Rk))
                    break;
                CloudAlignmentEvaluator::matMul3(Rk, Fa, Mn);
                float rm;
                buildRows(Mn, std::numeric_limits<float>::max(), rm);
                const float v = BottleneckEvaluator::computeFromCostMatrix(
                    cost, m, std::max(rm, lbSearchSq), std::numeric_limits<float>::max());
                if (!(v < bestSq))
                    break;
                std::vector<size_t> m2;
                if (!extractMatching(Mn, v, m2))
                    break;
                for (int k = 0; k < 9; ++k)
                    M[k] = Mn[k];
                bestSq = v;
                matching.swap(m2);
                ++iters;
                if (bestSq <= stopSq)
                    break;
            }
        }

        float R[9];
        CloudAlignmentEvaluator::rotationFromFrames(Fa, M, R);

        result.success = true;
        result.bestBottleneck = std::sqrt(bestSq);
        result.bestB1 = idxB[b1];
        result.bestB2 = idxB[b2];
        result.refineIterationsUsed = iters;
        for (int k = 0; k < 9; ++k)
            result.rotation[k] = R[k];

        result.matching.resize(m);
        for (size_t i = 0; i < m; ++i)
            result.matching[i] = haveMatching ? idxB[matching[i]] : 0;

        result.bestRotatedPointsA.resize(m * 3);
        for (size_t i = 0; i < m; ++i)
        {
            result.bestRotatedPointsA[i * 3 + 0] = M[0] * qx[i] + M[1] * qy[i] + M[2] * qz[i];
            result.bestRotatedPointsA[i * 3 + 1] = M[3] * qx[i] + M[4] * qy[i] + M[5] * qz[i];
            result.bestRotatedPointsA[i * 3 + 2] = M[6] * qx[i] + M[7] * qy[i] + M[8] * qz[i];
        }
        return result;
    }
};