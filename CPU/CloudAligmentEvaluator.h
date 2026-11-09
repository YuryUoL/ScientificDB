#pragma once
 
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <utility>
#include <cstdint>
#include "BottleneckEvaluator.h"
 
struct AlignmentResult {
    float bestBottleneck = std::numeric_limits<float>::max();
    size_t bestB1 = 0;
    size_t bestB2 = 0;
    float rotation[9] = {0};          // row-major, applied to centered cloud A
    float translation[3] = {0, 0, 0}; // p_B ~= R * p_A + t  (on raw, uncentered clouds)
    std::vector<float> bestRotatedPointsA; // xyz interleaved, in cloud B's space
    std::vector<size_t> matching;          // matching[i] = index in B matched to A[i]
    bool success = false;
};
 
class CloudAlignmentEvaluator {
public:
    struct Point3F {
        float x, y, z;
    };
 
    static inline float pointDist(const Point3F& p1, const Point3F& p2) {
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        float dz = p1.z - p2.z;
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }
 
    static inline float pointNorm(const Point3F& p) {
        return std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
    }
 
    /**
     * Symmetric Hausdorff distance. Valid lower bound on the bottleneck distance
     * (every A point needs a partner in B and every B point needs a partner in A).
     */
    static float computeHausdorffLowerBound(const std::vector<Point3F>& A, const std::vector<Point3F>& B) {
        auto directed = [](const std::vector<Point3F>& X, const std::vector<Point3F>& Y) {
            float worst = 0.0f;
            for (const auto& x : X) {
                float m = std::numeric_limits<float>::max();
                for (const auto& y : Y) m = std::min(m, pointDist(x, y));
                worst = std::max(worst, m);
            }
            return worst;
        };
        return std::max(directed(A, B), directed(B, A));
    }
 
    /// Picks the pair (a1, a2) maximizing the area of triangle (origin, a1, a2).
    static bool findBestTriangleA(const std::vector<Point3F>& pointsA, size_t& outA1, size_t& outA2) {
        size_t n = pointsA.size();
        if (n < 2) return false;
 
        float maxNormSq = 0.0f;
        for (const auto& p : pointsA) {
            maxNormSq = std::max(maxNormSq, p.x * p.x + p.y * p.y + p.z * p.z);
        }
 
        float maxAreaSq = -1.0f;
        outA1 = 0;
        outA2 = 1;
 
        for (size_t i = 0; i < n; ++i) {
            const auto& p1 = pointsA[i];
            for (size_t j = i + 1; j < n; ++j) {
                const auto& p2 = pointsA[j];
                float cx = p1.y * p2.z - p1.z * p2.y;
                float cy = p1.z * p2.x - p1.x * p2.z;
                float cz = p1.x * p2.y - p1.y * p2.x;
                float areaSq = cx * cx + cy * cy + cz * cz;
 
                if (areaSq > maxAreaSq) {
                    maxAreaSq = areaSq;
                    outA1 = i;
                    outA2 = j;
                }
            }
        }
        // Scale-relative degeneracy test
        return maxAreaSq > 1e-10f * maxNormSq * maxNormSq && maxAreaSq > 0.0f;
    }
 
    /**
     * Builds an orthonormal right-handed frame from (p1, p2).
     * Columns of F (row-major storage): e1 = p1/|p1|, e2 = e3 x e1, e3 = (p1 x p2)/|p1 x p2|.
     * Returns false if p1 or p2 are (nearly) collinear with the origin.
     */
    static bool makeFrame(const Point3F& p1, const Point3F& p2, float F[9]) {
        float l1 = pointNorm(p1);
        if (l1 < 1e-12f) return false;
        float e1x = p1.x / l1, e1y = p1.y / l1, e1z = p1.z / l1;
 
        float nx = p1.y * p2.z - p1.z * p2.y;
        float ny = p1.z * p2.x - p1.x * p2.z;
        float nz = p1.x * p2.y - p1.y * p2.x;
        float ln = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (ln < 1e-12f) return false;
        float e3x = nx / ln, e3y = ny / ln, e3z = nz / ln;
 
        float e2x = e3y * e1z - e3z * e1y;
        float e2y = e3z * e1x - e3x * e1z;
        float e2z = e3x * e1y - e3y * e1x;
 
        F[0] = e1x; F[1] = e2x; F[2] = e3x;
        F[3] = e1y; F[4] = e2y; F[5] = e3y;
        F[6] = e1z; F[7] = e2z; F[8] = e3z;
        return true;
    }
 
    /// R = Fb * Fa^T : proper rotation mapping frame(a1,a2) onto frame(b1,b2).
    static void rotationFromFrames(const float Fa[9], const float Fb[9], float R[9]) {
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                R[r * 3 + c] = Fb[r * 3 + 0] * Fa[c * 3 + 0]
                             + Fb[r * 3 + 1] * Fa[c * 3 + 1]
                             + Fb[r * 3 + 2] * Fa[c * 3 + 2];
            }
        }
    }
 
    static bool extractMatching(const float* costMatrix, size_t N, float threshold, std::vector<size_t>& matchingOut) {
        uint64_t adjMasks[BottleneckEvaluator::MAX_VERTICES];
        for (size_t i = 0; i < N; ++i) {
            uint64_t mask = 0;
            const float* row = &costMatrix[i * N];
            for (size_t j = 0; j < N; ++j) {
                if (row[j] <= threshold) {
                    mask |= (1ULL << j);
                }
            }
            adjMasks[i] = mask;
            if (mask == 0) return false;
        }
 
        std::vector<uint8_t> matchR(N, 255);
 
        auto findAugmentingPath = [&](size_t u, auto& self, uint64_t& visitedR) -> bool {
            uint64_t candidates;
            // Re-mask each iteration so right nodes visited by deeper recursion are skipped
            while ((candidates = adjMasks[u] & ~visitedR) != 0) {
                int v = __builtin_ctzll(candidates);
                visitedR |= (1ULL << v);
 
                if (matchR[v] == 255 || self(matchR[v], self, visitedR)) {
                    matchR[v] = static_cast<uint8_t>(u);
                    return true;
                }
            }
            return false;
        };
 
        for (size_t i = 0; i < N; ++i) {
            uint64_t visitedR = 0;
            if (!findAugmentingPath(i, findAugmentingPath, visitedR)) {
                return false;
            }
        }
 
        matchingOut.assign(N, 0);
        for (size_t v = 0; v < N; ++v) {
            if (matchR[v] != 255) {
                matchingOut[matchR[v]] = v;
            }
        }
        return true;
    }
 
    static AlignmentResult alignAndEvaluate(
        const std::vector<Point3F>& rawCloudA,
        const std::vector<Point3F>& rawCloudB,
        const std::vector<float>& pddA,
        const std::vector<float>& pddB)
    {
        AlignmentResult result;
        const size_t nA = rawCloudA.size();
        const size_t nB = rawCloudB.size();
        if (nA < 2 || nA != nB) return result;
        if (nA > BottleneckEvaluator::MAX_VERTICES) return result;
        if (pddA.size() < nA * nA || pddB.size() < nB * nB) return result;
 
        // 1. Center both clouds
        Point3F centroidA = {0, 0, 0}, centroidB = {0, 0, 0};
        for (const auto& p : rawCloudA) { centroidA.x += p.x; centroidA.y += p.y; centroidA.z += p.z; }
        for (const auto& p : rawCloudB) { centroidB.x += p.x; centroidB.y += p.y; centroidB.z += p.z; }
        const float invA = 1.0f / static_cast<float>(nA);
        const float invB = 1.0f / static_cast<float>(nB);
        centroidA.x *= invA; centroidA.y *= invA; centroidA.z *= invA;
        centroidB.x *= invB; centroidB.y *= invB; centroidB.z *= invB;
 
        std::vector<Point3F> cloudA(nA), cloudB(nB);
        for (size_t i = 0; i < nA; ++i)
            cloudA[i] = {rawCloudA[i].x - centroidA.x, rawCloudA[i].y - centroidA.y, rawCloudA[i].z - centroidA.z};
        for (size_t i = 0; i < nB; ++i)
            cloudB[i] = {rawCloudB[i].x - centroidB.x, rawCloudB[i].y - centroidB.y, rawCloudB[i].z - centroidB.z};
 
        // 2. Least degenerate triangle (origin, a1, a2) in A
        size_t a1, a2;
        if (!findBestTriangleA(cloudA, a1, a2)) return result;
 
        float Fa[9];
        if (!makeFrame(cloudA[a1], cloudA[a2], Fa)) return result;
 
        // 3. Rank candidate ordered pairs (b1, b2) by rotation-invariant features.
        //    Features are used ONLY for ordering, never as pruning bounds.
        const float na1 = pointNorm(cloudA[a1]);
        const float na2 = pointNorm(cloudA[a2]);
        const float dA12 = pointDist(cloudA[a1], cloudA[a2]);
 
        auto pddDiffRow = [&](const float* rowA, const float* rowB) {
            float m = 0.0f;
            for (size_t k = 0; k < nA; ++k) {
                m = std::max(m, std::abs(std::sqrt(rowA[k]) - std::sqrt(rowB[k])));
            }
            return m;
        };
 
        std::vector<float> pd1(nB), pd2(nB), nb(nB);
        for (size_t i = 0; i < nB; ++i) {
            pd1[i] = pddDiffRow(&pddA[a1 * nA], &pddB[i * nB]);
            pd2[i] = pddDiffRow(&pddA[a2 * nA], &pddB[i * nB]);
            nb[i]  = pointNorm(cloudB[i]);
        }
 
        struct CandidatePair {
            size_t b1, b2;
            float heuristicCost;
        };
 
        std::vector<CandidatePair> candidates;
        candidates.reserve(nB * (nB - 1));
 
        for (size_t i = 0; i < nB; ++i) {
            for (size_t j = 0; j < nB; ++j) {
                if (i == j) continue;
                float cost = std::abs(na1 - nb[i]) + std::abs(na2 - nb[j])
                           + pd1[i] + pd2[j]
                           + std::abs(dA12 - pointDist(cloudB[i], cloudB[j]));
                candidates.push_back({i, j, cost});
            }
        }
 
        std::sort(candidates.begin(), candidates.end(),
                  [](const CandidatePair& x, const CandidatePair& y) {
                      return x.heuristicCost < y.heuristicCost;
                  });
 
        // 4. Try candidates in order; prune with Hausdorff lower bound
        std::vector<Point3F> transformedA(nA);
        float costMatrix[BottleneckEvaluator::MAX_VERTICES * BottleneckEvaluator::MAX_VERTICES];
 
        for (const auto& cand : candidates) {
            float Fb[9], R[9];
            if (!makeFrame(cloudB[cand.b1], cloudB[cand.b2], Fb)) continue;
            rotationFromFrames(Fa, Fb, R);
 
            for (size_t i = 0; i < nA; ++i) {
                const auto& p = cloudA[i];
                transformedA[i] = {
                    R[0] * p.x + R[1] * p.y + R[2] * p.z,
                    R[3] * p.x + R[4] * p.y + R[5] * p.z,
                    R[6] * p.x + R[7] * p.y + R[8] * p.z
                };
            }
 
            const float hausdorffLB = computeHausdorffLowerBound(transformedA, cloudB);
            if (hausdorffLB >= result.bestBottleneck) continue;
 
            for (size_t i = 0; i < nA; ++i) {
                float* row = &costMatrix[i * nA];
                for (size_t j = 0; j < nB; ++j) {
                    row[j] = pointDist(transformedA[i], cloudB[j]);
                }
            }
 
            const float currentBottleneck = BottleneckEvaluator::computeFromCostMatrix(
                costMatrix, nA, hausdorffLB, result.bestBottleneck);
 
            if (currentBottleneck < result.bestBottleneck) {
                result.bestBottleneck = currentBottleneck;
                result.bestB1 = cand.b1;
                result.bestB2 = cand.b2;
                result.success = true;
 
                for (int k = 0; k < 9; ++k) result.rotation[k] = R[k];
 
                // p_B = R * (p_A - cA) + cB  =>  t = cB - R * cA
                float rotCx = R[0] * centroidA.x + R[1] * centroidA.y + R[2] * centroidA.z;
                float rotCy = R[3] * centroidA.x + R[4] * centroidA.y + R[5] * centroidA.z;
                float rotCz = R[6] * centroidA.x + R[7] * centroidA.y + R[8] * centroidA.z;
                result.translation[0] = centroidB.x - rotCx;
                result.translation[1] = centroidB.y - rotCy;
                result.translation[2] = centroidB.z - rotCz;
 
                result.bestRotatedPointsA.resize(nA * 3);
                for (size_t i = 0; i < nA; ++i) {
                    result.bestRotatedPointsA[i * 3 + 0] = transformedA[i].x + centroidB.x;
                    result.bestRotatedPointsA[i * 3 + 1] = transformedA[i].y + centroidB.y;
                    result.bestRotatedPointsA[i * 3 + 2] = transformedA[i].z + centroidB.z;
                }
 
                extractMatching(costMatrix, nA, currentBottleneck, result.matching);
            }
        }
 
        return result;
    }
};
