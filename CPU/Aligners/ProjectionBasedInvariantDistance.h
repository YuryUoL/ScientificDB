#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <cstddef>
#include "BottleneckEvaluator.h"

struct PbiFeasibilityResult
{
    bool feasible = false;              // at least one (a1, b1) has a valid inner matching < epsilon
    bool outerPerfectMatching = false;  // full 1-to-1 matching across all (a1, b1) < epsilon
    bool mirrored = false;              // true if the match was found against the mirrored B (needs allowReflection)
    size_t firstValidA1 = 0;
    size_t firstValidB1 = 0;
    std::vector<float> outerCostMatrix; // N x N row-major; 0.0f = matchable, max() = not matchable
};

struct PbiOptions
{
    float epsilon = 0.01f;              // relative if normalize == true (NPR), else absolute units
    bool allowSameAnchorSecond = false; // false: a2 != a1 and b2 != b1
    bool normalize = true;              // NPR: TG and rows 1-2 / R^2, last row / R^3
    bool allowReflection = false;       // false: chirality-sensitive (proper rotations only)
                                        // true : also accept B mirrored (one consistent global mirror)
};

class ProjectionBasedInvariantDistance
{
public:
    struct Point3F
    {
        float x, y, z;
    };

    /// `numCols` columns, each of length `colDim`, stored as col * colDim + d.
    struct ProjectionColumns
    {
        size_t numCols = 0;
        size_t colDim = 0;
        std::vector<float> data;

        const float* colPtr(size_t c) const { return &data[c * colDim]; }
    };

    /// Point-based representation PR(A; p1, p2) for n = 3:
    ///   tg   = (p1.p1, p1.p2, p2.p2)
    ///   proj = columns (q.p1, q.p2, q.(p1 x p2)) for every q not in {p1, p2}
    struct PairDescriptor
    {
        float tg[3] = {0, 0, 0};
        ProjectionColumns proj;
    };

    static inline float strictLessThreshold(float eps)
    {
        if (eps <= 0.0f) return -std::numeric_limits<float>::infinity();
        return std::nextafter(eps, 0.0f);
    }

    static inline float lInf3(const float* a, const float* b)
    {
        return std::max({std::fabs(a[0] - b[0]), std::fabs(a[1] - b[1]), std::fabs(a[2] - b[2])});
    }

    /// Can columns of A be matched 1-to-1 to columns of B with L_inf distance < eps?
    static bool canMatchProjectionColumns(
        const ProjectionColumns& pA,
        const ProjectionColumns& pB,
        float eps,
        std::vector<float>& scratch)
    {
        if (pA.numCols == 0 || pA.numCols != pB.numCols || pA.colDim != 3 || pB.colDim != 3)
            return false;
        if (pA.numCols > BottleneckEvaluator::MAX_VERTICES)
            return false;

        const size_t k = pA.numCols;
        const float strictEps = strictLessThreshold(eps);
        scratch.assign(k * k, std::numeric_limits<float>::max());

        for (size_t i = 0; i < k; ++i)
        {
            const float* cA = pA.colPtr(i);
            bool rowHasEdge = false;
            for (size_t j = 0; j < k; ++j)
            {
                const float d = lInf3(cA, pB.colPtr(j));
                scratch[i * k + j] = d;
                if (d <= strictEps) rowHasEdge = true;
            }
            if (!rowHasEdge) return false;
        }
        return BottleneckEvaluator::hasPerfectMatching(scratch.data(), k, strictEps);
    }

    /// PR / NPR descriptor for the ordered base pair (idx1, idx2) on a centered cloud.
    /// invR2 / invR3 are 1 for the plain PR, 1/R^2 and 1/R^3 for NPR.
    /// No frame is built, so there is no degenerate case: p1 || p2 just gives a zero last row.
    static PairDescriptor buildPairDescriptor(
        const std::vector<Point3F>& cloud,
        size_t idx1,
        size_t idx2,
        float invR2,
        float invR3)
    {
        PairDescriptor desc;
        const Point3F& p1 = cloud[idx1];
        const Point3F& p2 = cloud[idx2];

        // p3_perp = p1 x p2 (unnormalised, as in Definition D.1 / D.3)
        const float cx = p1.y * p2.z - p1.z * p2.y;
        const float cy = p1.z * p2.x - p1.x * p2.z;
        const float cz = p1.x * p2.y - p1.y * p2.x;

        desc.tg[0] = (p1.x * p1.x + p1.y * p1.y + p1.z * p1.z) * invR2;
        desc.tg[1] = (p1.x * p2.x + p1.y * p2.y + p1.z * p2.z) * invR2;
        desc.tg[2] = (p2.x * p2.x + p2.y * p2.y + p2.z * p2.z) * invR2;

        const size_t m = cloud.size();
        desc.proj.numCols = 0;
        desc.proj.colDim = 3;
        desc.proj.data.reserve(m * 3);

        for (size_t q = 0; q < m; ++q)
        {
            if (q == idx1 || q == idx2) continue;
            const Point3F& pq = cloud[q];
            desc.proj.data.push_back((pq.x * p1.x + pq.y * p1.y + pq.z * p1.z) * invR2);
            desc.proj.data.push_back((pq.x * p2.x + pq.y * p2.y + pq.z * p2.z) * invR2);
            desc.proj.data.push_back((pq.x * cx + pq.y * cy + pq.z * cz) * invR3); // chirality row
            ++desc.proj.numCols;
        }
        return desc;
    }

    /// Full nested test (inner graph per (a1,b1), then outer graph over all (a1,b1)).
    /// The outer matching is ALWAYS computed. Chirality-sensitive unless opt.allowReflection.
    static PbiFeasibilityResult evaluateFeasible(
        const std::vector<Point3F>& rawA,
        const std::vector<Point3F>& rawB,
        const PbiOptions& opt = PbiOptions())
    {
        PbiFeasibilityResult res;
        const size_t n = rawA.size();
        if (n < 3 || n != rawB.size() || n > BottleneckEvaluator::MAX_VERTICES)
            return res;
        if (!(opt.epsilon > 0.0f))
            return res;

        // Center both clouds
        double ax = 0, ay = 0, az = 0, bx = 0, by = 0, bz = 0;
        for (size_t i = 0; i < n; ++i)
        {
            ax += rawA[i].x; ay += rawA[i].y; az += rawA[i].z;
            bx += rawB[i].x; by += rawB[i].y; bz += rawB[i].z;
        }
        std::vector<Point3F> A(n), B(n);
        for (size_t i = 0; i < n; ++i)
        {
            A[i] = {float(rawA[i].x - ax / n), float(rawA[i].y - ay / n), float(rawA[i].z - az / n)};
            B[i] = {float(rawB[i].x - bx / n), float(rawB[i].y - by / n), float(rawB[i].z - bz / n)};
        }

        PbiFeasibilityResult direct = evaluateCentered(A, B, opt);
        if (!opt.allowReflection || direct.outerPerfectMatching)
            return direct;

        // One consistent global mirror of B (x -> -x flips handedness)
        std::vector<Point3F> Bm = B;
        for (auto& p : Bm) p.x = -p.x;

        PbiFeasibilityResult mirror = evaluateCentered(A, Bm, opt);
        mirror.mirrored = true;

        if (mirror.outerPerfectMatching) return mirror;
        if (!direct.feasible && mirror.feasible) return mirror;
        return direct;
    }

private:
    static float cloudRadius(const std::vector<Point3F>& c)
    {
        float r2 = 0.0f;
        for (const auto& p : c)
            r2 = std::max(r2, p.x * p.x + p.y * p.y + p.z * p.z);
        return std::sqrt(r2);
    }

    /// Both clouds already centered.
    static PbiFeasibilityResult evaluateCentered(
        const std::vector<Point3F>& A,
        const std::vector<Point3F>& B,
        const PbiOptions& opt)
    {
        PbiFeasibilityResult res;
        const size_t n = A.size();
        const float largeCost = std::numeric_limits<float>::max();
        const float strictEps = strictLessThreshold(opt.epsilon);
        res.outerCostMatrix.assign(n * n, largeCost);

        float iA2 = 1.0f, iA3 = 1.0f, iB2 = 1.0f, iB3 = 1.0f;
        if (opt.normalize)
        {
            const float rA = cloudRadius(A);
            const float rB = cloudRadius(B);
            if (rA <= 1e-12f || rB <= 1e-12f) return res; // degenerate: all points at the center
            iA2 = 1.0f / (rA * rA); iA3 = iA2 / rA;
            iB2 = 1.0f / (rB * rB); iB3 = iB2 / rB;
        }

        // Pair descriptors
        std::vector<std::vector<PairDescriptor>> descA(n, std::vector<PairDescriptor>(n));
        std::vector<std::vector<PairDescriptor>> descB(n, std::vector<PairDescriptor>(n));
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
            {
                if (!opt.allowSameAnchorSecond && i == j) continue;
                descA[i][j] = buildPairDescriptor(A, i, j, iA2, iA3);
                descB[i][j] = buildPairDescriptor(B, i, j, iB2, iB3);
            }

        // Second-anchor candidates (same for every first anchor except for exclusion of itself)
        auto makeCands = [&](size_t first)
        {
            std::vector<size_t> c;
            c.reserve(n);
            for (size_t s = 0; s < n; ++s)
            {
                if (!opt.allowSameAnchorSecond && s == first) continue;
                c.push_back(s);
            }
            return c;
        };

        std::vector<float> scratch;

        // Inner graphs for each fixed (a1, b1) -> outer edge
        for (size_t a1 = 0; a1 < n; ++a1)
        {
            const std::vector<size_t> candA2 = makeCands(a1);
            if (candA2.empty()) continue;

            for (size_t b1 = 0; b1 < n; ++b1)
            {
                const std::vector<size_t> candB2 = makeCands(b1);
                if (candB2.size() != candA2.size()) continue;

                const size_t m = candA2.size();
                if (m > BottleneckEvaluator::MAX_VERTICES) continue;

                std::vector<float> innerCost(m * m, largeCost);
                bool allRowsHaveEdge = true;

                for (size_t u = 0; u < m && allRowsHaveEdge; ++u)
                {
                    const PairDescriptor& dA = descA[a1][candA2[u]];
                    bool rowOk = false;

                    for (size_t v = 0; v < m; ++v)
                    {
                        const PairDescriptor& dB = descB[b1][candB2[v]];

                        // 1) triangular Gram matrices
                        if (!(lInf3(dA.tg, dB.tg) < opt.epsilon)) continue;
                        // 2) rectangular Gram columns (incl. chirality row)
                        if (!canMatchProjectionColumns(dA.proj, dB.proj, opt.epsilon, scratch)) continue;

                        innerCost[u * m + v] = 0.0f;
                        rowOk = true;
                    }
                    if (!rowOk) allRowsHaveEdge = false;
                }
                if (!allRowsHaveEdge) continue;

                if (BottleneckEvaluator::hasPerfectMatching(innerCost.data(), m, strictEps))
                {
                    res.outerCostMatrix[a1 * n + b1] = 0.0f;
                    if (!res.feasible)
                    {
                        res.feasible = true;
                        res.firstValidA1 = a1;
                        res.firstValidB1 = b1;
                    }
                }
            }
        }

        // Outer perfect matching, always evaluated
        if (res.feasible)
            res.outerPerfectMatching =
                BottleneckEvaluator::hasPerfectMatching(res.outerCostMatrix.data(), n, strictEps);

        return res;
    }
};