#pragma once
// Standalone injective-bottleneck alignment A -> B (|A| <= |B|) for clouds that are ALREADY
// centered at the origin (no centering, no translation). Rotation only: p_B ~= R * p_A.
// Depends only on AtomicBottleneckEvaluator.h.
//
// Point type: any struct with float-convertible members x, y, z (templated, no conversion needed).
// Options type: any struct with the fields of AtomicAlignmentOptions (templated; the old
// AlignmentOptions works unchanged).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>
#include "AtomicBottleneckEvaluator.h"

struct AtomicAlignmentOptions
{
    float acceptDistance = 0.0f;   // stop at first bottleneck <= this (0 = off)
    size_t maxCandidates = 0;      // only try best-ranked K frame pairs (0 = all)
    int refineIterations = 0;      // Kabsch refit / rematch rounds
    bool requireSameSymbol = false;
    bool exact = true;             // true: search until provably optimal; false: first valid match
    const std::vector<std::string> *symbolsA = nullptr; // indexed by FULL-cloud indices
    const std::vector<std::string> *symbolsB = nullptr;

    bool set(const std::string &k, const std::string &v)
    {
        try
        {
            auto b = [&]() { return v == "1" || v == "on" || v == "true"; };
            if (k == "accept" || k == "accept_distance" || k == "threshold") acceptDistance = std::stof(v);
            else if (k == "max" || k == "max_candidates") maxCandidates = static_cast<size_t>(std::stoul(v));
            else if (k == "refine") refineIterations = std::stoi(v);
            else if (k == "symbol" || k == "same_symbol") requireSameSymbol = b();
            else if (k == "exact" || k == "optimal") exact = b();
            else return false;
            return true;
        }
        catch (...) { return false; }
    }
};

struct SubsetAlignmentResult
{
    float bestBottleneck = std::numeric_limits<float>::max(); // Euclidean (not squared)
    size_t bestB1 = 0;                     // index into the FULL cloud B
    size_t bestB2 = 0;
    float rotation[9] = {0};               // row-major, p_B ~= R * p_A
    std::vector<float> bestRotatedPointsA; // xyz interleaved, selected A in idxA order
    std::vector<size_t> matching;          // matching[i] = FULL-cloud-B index matched to cloudA[idxA[i]]
    bool success = false;
    int refineIterationsUsed = 0;
};

class AtomicCloudAlignmentEvaluator
{
public:
    static constexpr size_t MAXV = AtomicBottleneckEvaluator::MAX_VERTICES;

    template <class PointT, class OptT = AtomicAlignmentOptions>
    static SubsetAlignmentResult alignSubsetAndEvaluate(
        const std::vector<PointT> &cloudA, const std::vector<size_t> &idxA,
        const std::vector<PointT> &cloudB, const std::vector<size_t> &idxB,
        const OptT &opt = OptT())
    {
        using BE = AtomicBottleneckEvaluator;
        constexpr float INF = std::numeric_limits<float>::infinity();
        constexpr float FMAX = std::numeric_limits<float>::max();

        SubsetAlignmentResult result;
        const size_t m = idxA.size(), n = idxB.size();
        if (m < 2 || m > n || n > MAXV)
            return result;
        for (size_t i : idxA) if (i >= cloudA.size()) return result;
        for (size_t j : idxB) if (j >= cloudB.size()) return result;

        // ---- 1. Select points (taken as-is: clouds are assumed centered) ----
        std::vector<V3> A(m), B(n);
        for (size_t i = 0; i < m; ++i) A[i] = {float(cloudA[idxA[i]].x), float(cloudA[idxA[i]].y), float(cloudA[idxA[i]].z)};
        for (size_t j = 0; j < n; ++j) B[j] = {float(cloudB[idxB[j]].x), float(cloudB[idxB[j]].y), float(cloudB[idxB[j]].z)};

        const bool checkSymbols = opt.requireSameSymbol && opt.symbolsA && opt.symbolsB;
        auto symOK = [&](size_t i, size_t j) -> bool
        { return (*opt.symbolsA)[idxA[i]] == (*opt.symbolsB)[idxB[j]]; };

        // ---- 2. Reference pair in A (max triangle area with origin) and its frame ----
        size_t a1, a2;
        if (!bestPairA(A, a1, a2)) return result;
        float Fa[9];
        if (!makeFrame(A[a1], A[a2], Fa)) return result;

        // ---- 3. Precomputation ----
        // q = Fa^T * a  (A expressed in its own frame); rotated A = Fb * q
        std::array<float, MAXV> qx{}, qy{}, qz{}, bx{}, by{}, bz{}, na2{}, nb2{};
        for (size_t i = 0; i < m; ++i)
        {
            const V3 &p = A[i];
            qx[i] = Fa[0] * p.x + Fa[3] * p.y + Fa[6] * p.z;
            qy[i] = Fa[1] * p.x + Fa[4] * p.y + Fa[7] * p.z;
            qz[i] = Fa[2] * p.x + Fa[5] * p.y + Fa[8] * p.z;
            na2[i] = p.x * p.x + p.y * p.y + p.z * p.z;
        }
        for (size_t j = 0; j < n; ++j)
        {
            bx[j] = B[j].x; by[j] = B[j].y; bz[j] = B[j].z;
            nb2[j] = bx[j] * bx[j] + by[j] * by[j] + bz[j] * bz[j];
        }

        // Rotation preserves norms => injective 1D bottleneck of sorted norms is a lower bound.
        float lbGlobal = 0.0f, scale = 0.0f;
        {
            std::vector<float> sa(m), sb(n);
            for (size_t i = 0; i < m; ++i) sa[i] = std::sqrt(na2[i]);
            for (size_t j = 0; j < n; ++j) sb[j] = std::sqrt(nb2[j]);
            std::sort(sa.begin(), sa.end());
            std::sort(sb.begin(), sb.end());
            lbGlobal = BE::injectiveBottleneck1D(sa.data(), m, sb.data(), n);
            scale = std::max(sa[m - 1], sb[n - 1]);
        }
        const float lbT = std::max(0.0f, lbGlobal - 1e-5f * scale);
        const float lbSearchSq = lbT * lbT;
        float stopSq = FMAX;
        if (opt.acceptDistance > 0.0f) stopSq = opt.acceptDistance * opt.acceptDistance;
        else if (opt.exact) stopSq = lbGlobal * lbGlobal;

        // Row order: farthest A points first (fail fast).
        std::array<uint8_t, MAXV> order{};
        for (size_t i = 0; i < m; ++i) order[i] = static_cast<uint8_t>(i);
        std::sort(order.begin(), order.begin() + m, [&](uint8_t x, uint8_t y) { return na2[x] > na2[y]; });

        // ---- 4. Rank ordered candidate pairs (b1,b2) in B ----
        const float s1 = na2[a1], s2 = na2[a2];
        const float dotA = A[a1].x * A[a2].x + A[a1].y * A[a2].y + A[a1].z * A[a2].z;
        const float d12A = s1 + s2 - 2.0f * dotA;
        const float areaA = std::sqrt(std::max(0.0f, s1 * s2 - dotA * dotA));

        struct Cand { float cost; uint16_t b1, b2; };
        std::vector<Cand> cands;
        cands.reserve(n * (n - 1));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
            {
                const float dot = bx[i] * bx[j] + by[i] * by[j] + bz[i] * bz[j];
                const float crossSq = nb2[i] * nb2[j] - dot * dot;
                if (crossSq <= 1e-10f * nb2[i] * nb2[j]) continue;
                const float d12B = nb2[i] + nb2[j] - 2.0f * dot;
                const float common = std::abs(d12A - d12B) + std::abs(areaA - std::sqrt(crossSq));
                if (!checkSymbols || (symOK(a1, i) && symOK(a2, j)))
                    cands.push_back({common + std::abs(s1 - nb2[i]) + std::abs(s2 - nb2[j]), (uint16_t)i, (uint16_t)j});
                if (!checkSymbols || (symOK(a2, i) && symOK(a1, j)))
                    cands.push_back({common + std::abs(s1 - nb2[j]) + std::abs(s2 - nb2[i]), (uint16_t)j, (uint16_t)i});
            }
        if (cands.empty()) return result;

        auto byCost = [](const Cand &x, const Cand &y) { return x.cost < y.cost; };
        if (opt.maxCandidates > 0 && opt.maxCandidates < cands.size())
        {
            std::partial_sort(cands.begin(), cands.begin() + opt.maxCandidates, cands.end(), byCost);
            cands.resize(opt.maxCandidates);
        }
        else
            std::sort(cands.begin(), cands.end(), byCost);

        // ---- 5. Evaluate candidates ----
        float cost[MAXV * MAXV]; // m x n, row stride n; forbidden pairs = +inf
        uint64_t adj[MAXV];

        // Fills cost rows for rotation Fb*q, adjacency masks for threshold t.
        // Returns false as soon as some row has no entry <= t.
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
                float mn = INF;
                uint64_t mask = 0;
                for (size_t j = 0; j < n; ++j)
                {
                    if (checkSymbols && !symOK(i, j)) { row[j] = INF; continue; }
                    const float dx = px - bx[j], dy = py - by[j], dz = pz - bz[j];
                    const float d = dx * dx + dy * dy + dz * dz;
                    row[j] = d;
                    mn = std::min(mn, d);
                    if (d <= t) mask |= (1ULL << j);
                }
                if (mn > t) return false;
                rowMinMax = std::max(rowMinMax, mn);
                adj[i] = mask;
            }
            return true;
        };

        float bestSq = FMAX;
        size_t bestC = 0;
        bool found = false;

        for (size_t c = 0; c < cands.size(); ++c)
        {
            float Fb[9];
            if (!makeFrame(B[cands[c].b1], B[cands[c].b2], Fb)) continue;

            const float t = found ? std::nextafter(bestSq, 0.0f) : FMAX;
            float rowMinMax;
            if (!buildRows(Fb, t, rowMinMax)) continue;
            if (!BE::matchInjective(adj, m, nullptr)) continue;

            const float lb = std::max(rowMinMax, lbSearchSq);
            const float v = BE::bottleneckInjective(cost, m, n, std::min(lb, t), t);
            if (v < bestSq)
            {
                bestSq = v; bestC = c; found = true;
                if (!opt.exact && opt.acceptDistance == 0.0f) break;
                if (bestSq <= stopSq) break;
            }
        }
        if (!found) return result;

        // ---- 6. Materialize winner & refine ----
        const size_t b1 = cands[bestC].b1, b2 = cands[bestC].b2;
        float M[9]; // M = Fb of the winner; rotated A = M * q
        if (!makeFrame(B[b1], B[b2], M)) return result;

        auto extractMatching = [&](const float Mx[9], float v, std::vector<size_t> &out) -> bool
        {
            float rm, t = v;
            if (!buildRows(Mx, t, rm))
            {
                t = v * (1.0f + 1e-5f) + 1e-30f;
                if (!buildRows(Mx, t, rm)) return false;
            }
            uint8_t matchR[MAXV];
            if (!BE::matchInjective(adj, m, matchR)) return false;
            out.assign(m, 0);
            for (size_t col = 0; col < n; ++col)
                if (matchR[col] != 255 && matchR[col] < m) out[matchR[col]] = col;
            return true;
        };

        std::vector<size_t> matching;
        const bool haveMatching = extractMatching(M, bestSq, matching);

        int iters = 0;
        if (opt.refineIterations > 0 && haveMatching && bestSq > stopSq)
        {
            for (int it = 0; it < opt.refineIterations; ++it)
            {
                float Rk[9], Mn[9], rm;
                if (!kabsch(A, B, matching, Rk)) break;
                matMul3(Rk, Fa, Mn); // rotated A = Mn*q = Rk*a
                if (!buildRows(Mn, FMAX, rm)) break;
                const float v = BE::bottleneckInjective(cost, m, n, std::max(rm, lbSearchSq), FMAX);
                if (!(v < bestSq)) break;
                std::vector<size_t> m2;
                if (!extractMatching(Mn, v, m2)) break;
                std::copy(Mn, Mn + 9, M);
                bestSq = v;
                matching.swap(m2);
                ++iters;
                if (bestSq <= stopSq) break;
            }
        }

        float R[9];
        rotationFromFrames(Fa, M, R);

        result.success = true;
        result.bestBottleneck = std::sqrt(bestSq);
        result.bestB1 = idxB[b1];
        result.bestB2 = idxB[b2];
        result.refineIterationsUsed = iters;
        std::copy(R, R + 9, result.rotation);

        result.matching.assign(m, 0);
        if (haveMatching)
            for (size_t i = 0; i < m; ++i) result.matching[i] = idxB[matching[i]];

        result.bestRotatedPointsA.resize(m * 3);
        for (size_t i = 0; i < m; ++i)
        {
            result.bestRotatedPointsA[i * 3 + 0] = M[0] * qx[i] + M[1] * qy[i] + M[2] * qz[i];
            result.bestRotatedPointsA[i * 3 + 1] = M[3] * qx[i] + M[4] * qy[i] + M[5] * qz[i];
            result.bestRotatedPointsA[i * 3 + 2] = M[6] * qx[i] + M[7] * qy[i] + M[8] * qz[i];
        }
        return result;
    }

private:
    struct V3 { float x, y, z; };

    /// Pair (i,j) maximizing |a_i x a_j| (triangle with the origin).
    static bool bestPairA(const std::vector<V3> &P, size_t &o1, size_t &o2)
    {
        const size_t n = P.size();
        if (n < 2) return false;
        float maxNormSq = 0.0f;
        for (const auto &p : P) maxNormSq = std::max(maxNormSq, p.x * p.x + p.y * p.y + p.z * p.z);
        float maxAreaSq = -1.0f;
        o1 = 0; o2 = 1;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = i + 1; j < n; ++j)
            {
                const float cx = P[i].y * P[j].z - P[i].z * P[j].y;
                const float cy = P[i].z * P[j].x - P[i].x * P[j].z;
                const float cz = P[i].x * P[j].y - P[i].y * P[j].x;
                const float a = cx * cx + cy * cy + cz * cz;
                if (a > maxAreaSq) { maxAreaSq = a; o1 = i; o2 = j; }
            }
        return maxAreaSq > 1e-10f * maxNormSq * maxNormSq && maxAreaSq > 0.0f;
    }

    /// Right-handed orthonormal frame (columns e1,e2,e3; row-major) from p1 (-> e1) and p2 (in e1-e2 plane).
    static bool makeFrame(const V3 &p1, const V3 &p2, float F[9])
    {
        const float l1 = std::sqrt(p1.x * p1.x + p1.y * p1.y + p1.z * p1.z);
        if (l1 < 1e-12f) return false;
        const float e1x = p1.x / l1, e1y = p1.y / l1, e1z = p1.z / l1;
        const float nx = p1.y * p2.z - p1.z * p2.y;
        const float ny = p1.z * p2.x - p1.x * p2.z;
        const float nz = p1.x * p2.y - p1.y * p2.x;
        const float ln = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (ln < 1e-12f) return false;
        const float e3x = nx / ln, e3y = ny / ln, e3z = nz / ln;
        const float e2x = e3y * e1z - e3z * e1y;
        const float e2y = e3z * e1x - e3x * e1z;
        const float e2z = e3x * e1y - e3y * e1x;
        F[0] = e1x; F[1] = e2x; F[2] = e3x;
        F[3] = e1y; F[4] = e2y; F[5] = e3y;
        F[6] = e1z; F[7] = e2z; F[8] = e3z;
        return true;
    }

    static void rotationFromFrames(const float Fa[9], const float Fb[9], float R[9]) // R = Fb * Fa^T
    {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R[r * 3 + c] = Fb[r * 3] * Fa[c * 3] + Fb[r * 3 + 1] * Fa[c * 3 + 1] + Fb[r * 3 + 2] * Fa[c * 3 + 2];
    }

    static void matMul3(const float X[9], const float Y[9], float Z[9])
    {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                Z[r * 3 + c] = X[r * 3] * Y[c] + X[r * 3 + 1] * Y[3 + c] + X[r * 3 + 2] * Y[6 + c];
    }

    /// Best rotation R (no translation) minimizing sum |R a_i - b_match[i]|^2 (Horn quaternion method).
    static bool kabsch(const std::vector<V3> &A, const std::vector<V3> &B,
                       const std::vector<size_t> &match, float Rout[9])
    {
        double S[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        for (size_t i = 0; i < A.size(); ++i)
        {
            const double a[3] = {A[i].x, A[i].y, A[i].z};
            const V3 &bp = B[match[i]];
            const double b[3] = {bp.x, bp.y, bp.z};
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    S[r][c] += a[r] * b[c];
        }
        double N[4][4] = {
            {S[0][0] + S[1][1] + S[2][2], S[1][2] - S[2][1], S[2][0] - S[0][2], S[0][1] - S[1][0]},
            {S[1][2] - S[2][1], S[0][0] - S[1][1] - S[2][2], S[0][1] + S[1][0], S[2][0] + S[0][2]},
            {S[2][0] - S[0][2], S[0][1] + S[1][0], -S[0][0] + S[1][1] - S[2][2], S[1][2] + S[2][1]},
            {S[0][1] - S[1][0], S[2][0] + S[0][2], S[1][2] + S[2][1], -S[0][0] - S[1][1] + S[2][2]}};
        double V[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
        for (int sweep = 0; sweep < 60; ++sweep)
        {
            double off = 0, tot = 0;
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 4; ++j)
                {
                    tot += N[i][j] * N[i][j];
                    if (i != j) off += N[i][j] * N[i][j];
                }
            if (off <= 1e-26 * tot) break;
            for (int p = 0; p < 3; ++p)
                for (int q = p + 1; q < 4; ++q)
                {
                    if (N[p][q] == 0.0) continue;
                    const double theta = (N[q][q] - N[p][p]) / (2.0 * N[p][q]);
                    const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                    const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                    for (int k = 0; k < 4; ++k) { const double kp = N[k][p], kq = N[k][q]; N[k][p] = c * kp - s * kq; N[k][q] = s * kp + c * kq; }
                    for (int k = 0; k < 4; ++k) { const double pk = N[p][k], qk = N[q][k]; N[p][k] = c * pk - s * qk; N[q][k] = s * pk + c * qk; }
                    for (int k = 0; k < 4; ++k) { const double kp = V[k][p], kq = V[k][q]; V[k][p] = c * kp - s * kq; V[k][q] = s * kp + c * kq; }
                }
        }
        int best = 0;
        for (int i = 1; i < 4; ++i) if (N[i][i] > N[best][best]) best = i;
        double w = V[0][best], x = V[1][best], y = V[2][best], z = V[3][best];
        const double len = std::sqrt(w * w + x * x + y * y + z * z);
        if (!(len > 0.0)) return false;
        w /= len; x /= len; y /= len; z /= len;
        Rout[0] = float(w * w + x * x - y * y - z * z);
        Rout[1] = float(2 * (x * y - w * z));
        Rout[2] = float(2 * (x * z + w * y));
        Rout[3] = float(2 * (x * y + w * z));
        Rout[4] = float(w * w - x * x + y * y - z * z);
        Rout[5] = float(2 * (y * z - w * x));
        Rout[6] = float(2 * (x * z - w * y));
        Rout[7] = float(2 * (y * z + w * x));
        Rout[8] = float(w * w - x * x - y * y + z * z);
        return true;
    }
};