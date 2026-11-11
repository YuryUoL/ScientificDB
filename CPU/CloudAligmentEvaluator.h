#pragma once

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
#include <utility>
#include <cstdint>
#include <cstddef>
#include <string>
#include "BottleneckEvaluator.h"

struct AlignmentResult
{
    float bestBottleneck = std::numeric_limits<float>::max();
    size_t bestB1 = 0;
    size_t bestB2 = 0;
    float rotation[9] = {0};               // row-major, applied to centered cloud A
    float translation[3] = {0, 0, 0};      // p_B ~= R * p_A + t  (on raw, uncentered clouds)
    std::vector<float> bestRotatedPointsA; // xyz interleaved, in cloud B's space
    std::vector<size_t> matching;          // matching[i] = index in B matched to A[i]
    bool success = false;
    int refineIterationsUsed = 0; // accepted refinement steps (0 if refinement off / no gain)
};

struct AlignmentOptions
{
    /// Stop as soon as a bottleneck <= this distance is found (0 = only stop when provably optimal).
    float acceptDistance = 0.0f;
    /// Only try the best-ranked K candidate pairs (0 = all).
    size_t maxCandidates = 0;
    /// Max rounds of "Kabsch re-fit on all points -> re-match -> keep if bottleneck strictly improves".
    int refineIterations = 0;
    /// If true, restrict bipartite matching and frame generation so that element symbols match.
    bool requireSameSymbol = false;
    /// True = find provably optimal minimum (default: 1); False = return any valid match / early-exit (0).
    bool exact = true;

    const std::vector<std::string> *symbolsA = nullptr;
    const std::vector<std::string> *symbolsB = nullptr;

    bool set(const std::string &k, const std::string &v)
    {
        try
        {
            auto b = [&]()
            { return v == "1" || v == "on" || v == "true"; };
            if (k == "accept" || k == "accept_distance" || k == "threshold")
                acceptDistance = std::stof(v);
            else if (k == "max" || k == "max_candidates")
                maxCandidates = static_cast<size_t>(std::stoul(v));
            else if (k == "refine")
                refineIterations = std::stoi(v);
            else if (k == "symbol" || k == "same_symbol")
                requireSameSymbol = b();
            else if (k == "exact" || k == "optimal")
                exact = b();
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

class CloudAlignmentEvaluator
{
public:
    struct Point3F
    {
        float x, y, z;
    };

    static inline float pointDist(const Point3F &p1, const Point3F &p2)
    {
        float dx = p1.x - p2.x, dy = p1.y - p2.y, dz = p1.z - p2.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    static inline float pointNorm(const Point3F &p)
    {
        return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    }

    /// Picks the pair (a1, a2) maximizing the area of triangle (origin, a1, a2).
    static bool findBestTriangleA(const std::vector<Point3F> &pointsA, size_t &outA1, size_t &outA2)
    {
        size_t n = pointsA.size();
        if (n < 2)
            return false;

        float maxNormSq = 0.0f;
        for (const auto &p : pointsA)
            maxNormSq = std::max(maxNormSq, p.x * p.x + p.y * p.y + p.z * p.z);

        float maxAreaSq = -1.0f;
        outA1 = 0;
        outA2 = 1;
        for (size_t i = 0; i < n; ++i)
        {
            const auto &p1 = pointsA[i];
            for (size_t j = i + 1; j < n; ++j)
            {
                const auto &p2 = pointsA[j];
                float cx = p1.y * p2.z - p1.z * p2.y;
                float cy = p1.z * p2.x - p1.x * p2.z;
                float cz = p1.x * p2.y - p1.y * p2.x;
                float areaSq = cx * cx + cy * cy + cz * cz;
                if (areaSq > maxAreaSq)
                {
                    maxAreaSq = areaSq;
                    outA1 = i;
                    outA2 = j;
                }
            }
        }
        return maxAreaSq > 1e-10f * maxNormSq * maxNormSq && maxAreaSq > 0.0f;
    }

    /// Orthonormal right-handed frame (columns e1,e2,e3; row-major storage) from (p1, p2).
    static bool makeFrame(const Point3F &p1, const Point3F &p2, float F[9])
    {
        float l1 = pointNorm(p1);
        if (l1 < 1e-12f)
            return false;
        float e1x = p1.x / l1, e1y = p1.y / l1, e1z = p1.z / l1;

        float nx = p1.y * p2.z - p1.z * p2.y;
        float ny = p1.z * p2.x - p1.x * p2.z;
        float nz = p1.x * p2.y - p1.y * p2.x;
        float ln = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (ln < 1e-12f)
            return false;
        float e3x = nx / ln, e3y = ny / ln, e3z = nz / ln;

        float e2x = e3y * e1z - e3z * e1y;
        float e2y = e3z * e1x - e3x * e1z;
        float e2z = e3x * e1y - e3y * e1x;

        F[0] = e1x;
        F[1] = e2x;
        F[2] = e3x;
        F[3] = e1y;
        F[4] = e2y;
        F[5] = e3y;
        F[6] = e1z;
        F[7] = e2z;
        F[8] = e3z;
        return true;
    }

    /// R = Fb * Fa^T
    static void rotationFromFrames(const float Fa[9], const float Fb[9], float R[9])
    {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R[r * 3 + c] = Fb[r * 3 + 0] * Fa[c * 3 + 0] + Fb[r * 3 + 1] * Fa[c * 3 + 1] + Fb[r * 3 + 2] * Fa[c * 3 + 2];
    }

    static AlignmentResult alignAndEvaluate(
        const std::vector<Point3F> &rawCloudA,
        const std::vector<Point3F> &rawCloudB,
        const AlignmentOptions &opt = AlignmentOptions())
    {
        constexpr size_t MAXV = BottleneckEvaluator::MAX_VERTICES;
        AlignmentResult result;
        const size_t n = rawCloudA.size();
        if (n < 2 || n != rawCloudB.size() || n > MAXV)
            return result;

        // ---------- 1. Center both clouds ----------
        double ax = 0, ay = 0, az = 0, bxs = 0, bys = 0, bzs = 0;
        for (const auto &p : rawCloudA)
        {
            ax += p.x;
            ay += p.y;
            az += p.z;
        }
        for (const auto &p : rawCloudB)
        {
            bxs += p.x;
            bys += p.y;
            bzs += p.z;
        }
        const Point3F cA = {float(ax / n), float(ay / n), float(az / n)};
        const Point3F cB = {float(bxs / n), float(bys / n), float(bzs / n)};

        std::vector<Point3F> A(n), B(n);
        for (size_t i = 0; i < n; ++i)
        {
            A[i] = {rawCloudA[i].x - cA.x, rawCloudA[i].y - cA.y, rawCloudA[i].z - cA.z};
            B[i] = {rawCloudB[i].x - cB.x, rawCloudB[i].y - cB.y, rawCloudB[i].z - cB.z};
        }

        // ---------- 2. Reference triangle in A, frame Fa ----------
        size_t a1, a2;
        if (!findBestTriangleA(A, a1, a2))
            return result;
        float Fa[9];
        if (!makeFrame(A[a1], A[a2], Fa))
            return result;

        // ---------- 3. Precomputation ----------
        std::array<float, MAXV> qx, qy, qz, bx, by, bz, na2, nb2;
        for (size_t i = 0; i < n; ++i)
        {
            const auto &p = A[i];
            qx[i] = Fa[0] * p.x + Fa[3] * p.y + Fa[6] * p.z;
            qy[i] = Fa[1] * p.x + Fa[4] * p.y + Fa[7] * p.z;
            qz[i] = Fa[2] * p.x + Fa[5] * p.y + Fa[8] * p.z;
            bx[i] = B[i].x;
            by[i] = B[i].y;
            bz[i] = B[i].z;
            na2[i] = p.x * p.x + p.y * p.y + p.z * p.z;
            nb2[i] = B[i].x * B[i].x + B[i].y * B[i].y + B[i].z * B[i].z;
        }

        float lbGlobal = 0.0f, scale = 0.0f;
        {
            std::vector<float> sa(n), sb(n);
            for (size_t i = 0; i < n; ++i)
            {
                sa[i] = std::sqrt(na2[i]);
                sb[i] = std::sqrt(nb2[i]);
            }
            std::sort(sa.begin(), sa.end());
            std::sort(sb.begin(), sb.end());
            for (size_t i = 0; i < n; ++i)
                lbGlobal = std::max(lbGlobal, std::abs(sa[i] - sb[i]));
            scale = std::max(sa[n - 1], sb[n - 1]);
        }
        const float lbSearchSq = std::pow(std::max(0.0f, lbGlobal - 1e-5f * scale), 2.0f);
        float stopSq = std::numeric_limits<float>::max();
        if (opt.acceptDistance > 0.0f)
        {
            stopSq = opt.acceptDistance * opt.acceptDistance;
        }
        else if (opt.exact)
        {
            stopSq = std::max(lbGlobal * lbGlobal, opt.acceptDistance * opt.acceptDistance);
        }
        else
        {
            stopSq = std::numeric_limits<float>::max();
        }

        std::array<uint8_t, MAXV> order;
        for (size_t i = 0; i < n; ++i)
            order[i] = static_cast<uint8_t>(i);
        std::sort(order.begin(), order.begin() + n, [&](uint8_t x, uint8_t y)
                  { return na2[x] > na2[y]; });

        const uint64_t fullMask = (n == 64) ? ~0ULL : ((1ULL << n) - 1ULL);

        // ---------- 4. Rank ordered pairs (b1,b2) with optional symbol constraint ----------
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

        bool checkSymbols = opt.requireSameSymbol && opt.symbolsA && opt.symbolsB;
        std::string symA1 = checkSymbols ? (*opt.symbolsA)[a1] : "";
        std::string symA2 = checkSymbols ? (*opt.symbolsA)[a2] : "";

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

                // Option 1: b1 matches a1, b2 matches a2
                if (!checkSymbols || ((*opt.symbolsB)[i] == symA1 && (*opt.symbolsB)[j] == symA2))
                {
                    cands.push_back({common + std::abs(s1 - nb2[i]) + std::abs(s2 - nb2[j]),
                                     static_cast<uint16_t>(i), static_cast<uint16_t>(j)});
                }

                // Option 2: b1 matches a2, b2 matches a1 (swapped)
                if (!checkSymbols || ((*opt.symbolsB)[i] == symA2 && (*opt.symbolsB)[j] == symA1))
                {
                    cands.push_back({common + std::abs(s1 - nb2[j]) + std::abs(s2 - nb2[i]),
                                     static_cast<uint16_t>(j), static_cast<uint16_t>(i)});
                }
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
        {
            std::sort(cands.begin(), cands.end(), byCost);
        }

        // ---------- 5. Evaluate candidates ----------
        float cost[MAXV * MAXV];
        uint64_t adj[MAXV];

        auto buildRows = [&](const float Fb[9], float t, float &rowMinMax) -> bool
        {
            uint64_t cover = 0;
            rowMinMax = 0.0f;
            for (size_t k = 0; k < n; ++k)
            {
                const size_t i = order[k];
                const float px = Fb[0] * qx[i] + Fb[1] * qy[i] + Fb[2] * qz[i];
                const float py = Fb[3] * qx[i] + Fb[4] * qy[i] + Fb[5] * qz[i];
                const float pz = Fb[6] * qx[i] + Fb[7] * qy[i] + Fb[8] * qz[i];
                float *row = &cost[i * n];
                float mn = std::numeric_limits<float>::max();
                for (size_t j = 0; j < n; ++j)
                {
                    if (opt.requireSameSymbol && opt.symbolsA && opt.symbolsB)
                    {
                        if ((*opt.symbolsA)[i] != (*opt.symbolsB)[j])
                        {
                            row[j] = std::numeric_limits<float>::max();
                            continue;
                        }
                    }
                    const float dx = px - bx[j], dy = py - by[j], dz = pz - bz[j];
                    const float d = dx * dx + dy * dy + dz * dz;
                    row[j] = d;
                    mn = std::min(mn, d);
                }
                if (mn > t)
                    return false;
                rowMinMax = std::max(rowMinMax, mn);
                uint64_t m = 0;
                for (size_t j = 0; j < n; ++j)
                    m |= (uint64_t)(row[j] <= t) << j;
                adj[i] = m;
                cover |= m;
            }
            return cover == fullMask;
        };

        float bestSq = std::numeric_limits<float>::max();
        size_t bestC = 0;
        bool found = false;

        for (size_t c = 0; c < cands.size(); ++c)
        {
            float Fb[9];
            if (!makeFrame(B[cands[c].b1], B[cands[c].b2], Fb))
                continue;

            const float t = found ? std::nextafter(bestSq, 0.0f) : std::numeric_limits<float>::max();
            float rowMinMax;
            if (!buildRows(Fb, t, rowMinMax))
                continue;
            if (!matchFromMasks(adj, n, nullptr))
                continue;

            const float lb = std::max(rowMinMax, lbSearchSq);
            const float v = BottleneckEvaluator::computeFromCostMatrix(cost, n, std::min(lb, t), t);
            if (v < bestSq) {
                bestSq = v;
                bestC = c;
                found = true;
                if (!opt.exact && opt.acceptDistance == 0.0f) {
                    break; // Early exit on first valid match when exact=0
                }
                if (bestSq <= stopSq) break;
            }
        }

        if (!found)
            return result;

        // ---------- 6. Materialize the winner (and optionally refine) ----------
        const size_t b1 = cands[bestC].b1, b2 = cands[bestC].b2;
        float M[9];
        if (!makeFrame(B[b1], B[b2], M))
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
            if (!matchFromMasks(adj, n, matchR))
                return false;
            out.assign(n, 0);
            for (size_t col = 0; col < n; ++col)
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
                if (!kabsch(A, B, matching, Rk))
                    break;
                matMul3(Rk, Fa, Mn);
                float rm;
                buildRows(Mn, std::numeric_limits<float>::max(), rm);
                const float v = BottleneckEvaluator::computeFromCostMatrix(
                    cost, n, std::max(rm, lbSearchSq), std::numeric_limits<float>::max());
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
        rotationFromFrames(Fa, M, R);

        result.success = true;
        result.bestBottleneck = std::sqrt(bestSq);
        result.bestB1 = b1;
        result.bestB2 = b2;
        result.matching = std::move(matching);
        result.refineIterationsUsed = iters;
        for (int k = 0; k < 9; ++k)
            result.rotation[k] = R[k];

        result.translation[0] = cB.x - (R[0] * cA.x + R[1] * cA.y + R[2] * cA.z);
        result.translation[1] = cB.y - (R[3] * cA.x + R[4] * cA.y + R[5] * cA.z);
        result.translation[2] = cB.z - (R[6] * cA.x + R[7] * cA.y + R[8] * cA.z);

        result.bestRotatedPointsA.resize(n * 3);
        for (size_t i = 0; i < n; ++i)
        {
            result.bestRotatedPointsA[i * 3 + 0] = M[0] * qx[i] + M[1] * qy[i] + M[2] * qz[i] + cB.x;
            result.bestRotatedPointsA[i * 3 + 1] = M[3] * qx[i] + M[4] * qy[i] + M[5] * qz[i] + cB.y;
            result.bestRotatedPointsA[i * 3 + 2] = M[6] * qx[i] + M[7] * qy[i] + M[8] * qz[i] + cB.z;
        }
        return result;
    }

private:
    static void matMul3(const float X[9], const float Y[9], float Z[9])
    {
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                Z[r * 3 + c] = X[r * 3] * Y[c] + X[r * 3 + 1] * Y[3 + c] + X[r * 3 + 2] * Y[6 + c];
    }

    static bool kabsch(const std::vector<Point3F> &A, const std::vector<Point3F> &B,
                       const std::vector<size_t> &match, float Rout[9])
    {
        double S[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
        for (size_t i = 0; i < A.size(); ++i)
        {
            const double a[3] = {A[i].x, A[i].y, A[i].z};
            const Point3F &bp = B[match[i]];
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
                    if (i != j)
                        off += N[i][j] * N[i][j];
                }
            if (off <= 1e-26 * tot)
                break;
            for (int p = 0; p < 3; ++p)
            {
                for (int q = p + 1; q < 4; ++q)
                {
                    if (N[p][q] == 0.0)
                        continue;
                    const double theta = (N[q][q] - N[p][p]) / (2.0 * N[p][q]);
                    const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                    const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                    for (int k = 0; k < 4; ++k)
                    {
                        const double kp = N[k][p], kq = N[k][q];
                        N[k][p] = c * kp - s * kq;
                        N[k][q] = s * kp + c * kq;
                    }
                    for (int k = 0; k < 4; ++k)
                    {
                        const double pk = N[p][k], qk = N[q][k];
                        N[p][k] = c * pk - s * qk;
                        N[q][k] = s * pk + c * qk;
                    }
                    for (int k = 0; k < 4; ++k)
                    {
                        const double kp = V[k][p], kq = V[k][q];
                        V[k][p] = c * kp - s * kq;
                        V[k][q] = s * kp + c * kq;
                    }
                }
            }
        }
        int best = 0;
        for (int i = 1; i < 4; ++i)
            if (N[i][i] > N[best][best])
                best = i;
        double w = V[0][best], x = V[1][best], y = V[2][best], z = V[3][best];
        const double len = std::sqrt(w * w + x * x + y * y + z * z);
        if (!(len > 0.0))
            return false;
        w /= len;
        x /= len;
        y /= len;
        z /= len;
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

    static bool matchFromMasks(const uint64_t *adj, size_t n, uint8_t *matchROut)
    {
        uint8_t local[BottleneckEvaluator::MAX_VERTICES];
        uint8_t *matchR = matchROut ? matchROut : local;
        for (size_t i = 0; i < n; ++i)
            matchR[i] = 255;
        uint64_t matched = 0;
        for (size_t i = 0; i < n; ++i)
        {
            uint64_t visited = 0;
            if (!augment(i, adj, matchR, visited, matched))
                return false;
        }
        return true;
    }

    static bool augment(size_t u, const uint64_t *adj, uint8_t *matchR, uint64_t &visited, uint64_t &matched)
    {
        uint64_t freeC = adj[u] & ~matched;
        if (freeC)
        {
            int v = __builtin_ctzll(freeC);
            matchR[v] = static_cast<uint8_t>(u);
            matched |= (1ULL << v);
            return true;
        }
        uint64_t c;
        while ((c = adj[u] & ~visited) != 0)
        {
            int v = __builtin_ctzll(c);
            visited |= (1ULL << v);
            if (augment(matchR[v], adj, matchR, visited, matched))
            {
                matchR[v] = static_cast<uint8_t>(u);
                return true;
            }
        }
        return false;
    }
};