#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <limits>

class BottleneckEvaluator {
public:
    static constexpr size_t MAX_VERTICES = 64;

    template <typename Point3D>
    static void computePDD(const std::vector<Point3D>& points, std::vector<float>& pddMatrix) {
        size_t n = points.size();
        pddMatrix.resize(n * n);

        for (size_t i = 0; i < n; ++i) {
            float* row = &pddMatrix[i * n];
            for (size_t j = 0; j < n; ++j) {
                float dx = points[i].x - points[j].x;
                float dy = points[i].y - points[j].y;
                float dz = points[i].z - points[j].z;
                row[j] = std::sqrt(dx * dx + dy * dy + dz * dz); // Store Euclidean once!
            }
            std::sort(row, row + n);
        }
    }

    static void computeCostMatrix(const float* pddA, const float* pddB, size_t N, float* costMatrixOut) {
        for (size_t i = 0; i < N; ++i) {
            const float* rowA = &pddA[i * N];
            for (size_t j = 0; j < N; ++j) {
                const float* rowB = &pddB[j * N];
                float maxDiff = 0.0f;
                for (size_t k = 1; k < N; ++k) { // k=0 is always 0.0f
                    float diff = std::fabs(rowA[k] - rowB[k]);
                    if (diff > maxDiff) maxDiff = diff;
                }
                costMatrixOut[i * N + j] = maxDiff;
            }
        }
    }

    /**
     * @brief Fast Stage-3 PDD filter: builds bitmask adjacency directly with early exit,
     *        skipping full costMatrix construction and binary search.
     */
    static bool passesPddThreshold(const float* pddA, const float* pddB, size_t N, float threshold) {
        if (N == 0 || N > MAX_VERTICES) return false;
        if (N == 1) return true;

        const uint64_t fullMask = (N == 64) ? ~0ULL : ((1ULL << N) - 1ULL);
        uint64_t adjMasks[MAX_VERTICES];
        uint64_t colCover = 0;

        for (size_t i = 0; i < N; ++i) {
            const float* rowA = &pddA[i * N];
            const float a1 = rowA[1];
            uint64_t mask = 0;

            for (size_t j = 0; j < N; ++j) {
                const float* rowB = &pddB[j * N];
                // Quick nearest-neighbor check before scanning full row
                if (std::fabs(a1 - rowB[1]) > threshold) continue;

                bool ok = true;
                for (size_t k = 2; k < N; ++k) {
                    if (std::fabs(rowA[k] - rowB[k]) > threshold) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    mask |= (1ULL << j);
                }
            }

            if (mask == 0) return false; // Row i has no valid match
            adjMasks[i] = mask;
            colCover |= mask;
        }

        if (colCover != fullMask) return false; // Hall's condition quick check: some column has 0 incident edges

        return matchFromMasks(adjMasks, N);
    }

    /**
     * @brief Computes exact Bottleneck Distance accelerated via pre-filtering to [lowBound, highBound].
     */
    static float computeFromCostMatrix(const float* costMatrix, size_t N,
                                       float lowBound = 0.0f,
                                       float highBound = std::numeric_limits<float>::max()) {
        if (N == 0) return 0.0f;
        if (N > MAX_VERTICES) {
            throw std::invalid_argument("BottleneckEvaluator: N exceeds MAX_VERTICES (64)");
        }
        if (lowBound >= highBound) return highBound;

        // Collect ONLY thresholds inside [lowBound, highBound] BEFORE sorting
        float thresholds[MAX_VERTICES * MAX_VERTICES];
        size_t count = 0;
        const size_t total = N * N;

        for (size_t i = 0; i < total; ++i) {
            float v = costMatrix[i];
            if (v >= lowBound && v <= highBound) {
                thresholds[count++] = v;
            }
        }

        if (count == 0) return highBound;

        std::sort(thresholds, thresholds + count);

        size_t uniqueCount = 0;
        for (size_t i = 0; i < count; ++i) {
            if (i == 0 || thresholds[i] != thresholds[i - 1]) {
                thresholds[uniqueCount++] = thresholds[i];
            }
        }

        int bestValidIdx = -1;
        int l = 0, h = static_cast<int>(uniqueCount) - 1;

        while (l <= h) {
            int mid = l + (h - l) / 2;
            if (hasPerfectMatching(costMatrix, N, thresholds[mid])) {
                bestValidIdx = mid;
                h = mid - 1;
            } else {
                l = mid + 1;
            }
        }

        return (bestValidIdx != -1) ? thresholds[bestValidIdx] : highBound;
    }

    static bool hasPerfectMatching(const float* costMatrix, size_t N, float threshold) {
        const uint64_t fullMask = (N == 64) ? ~0ULL : ((1ULL << N) - 1ULL);
        uint64_t adjMasks[MAX_VERTICES];
        uint64_t colCover = 0;

        for (size_t i = 0; i < N; ++i) {
            uint64_t mask = 0;
            const float* row = &costMatrix[i * N];
            for (size_t j = 0; j < N; ++j) {
                if (row[j] <= threshold) {
                    mask |= (1ULL << j);
                }
            }
            if (mask == 0) return false;
            adjMasks[i] = mask;
            colCover |= mask;
        }

        if (colCover != fullMask) return false;
        return matchFromMasks(adjMasks, N);
    }

private:
    static bool matchFromMasks(const uint64_t* adjMasks, size_t N) {
        uint8_t matchR[MAX_VERTICES];
        for (size_t i = 0; i < N; ++i) matchR[i] = 255;
        uint64_t matched = 0;

        for (size_t i = 0; i < N; ++i) {
            uint64_t visitedR = 0;
            if (!findAugmentingPath(i, adjMasks, matchR, visitedR, matched)) {
                return false;
            }
        }
        return true;
    }

    inline static bool findAugmentingPath(size_t u, const uint64_t* adjMasks, uint8_t* matchR,
                                          uint64_t& visitedR, uint64_t& matched) {
        // Greedy O(1) assignment if any adjacent vertex on the right is still unmatched
        uint64_t freeC = adjMasks[u] & ~matched;
        if (freeC) {
            int v = __builtin_ctzll(freeC);
            matchR[v] = static_cast<uint8_t>(u);
            matched |= (1ULL << v);
            return true;
        }

        uint64_t candidates = adjMasks[u] & ~visitedR;
        while (candidates) {
            int v = __builtin_ctzll(candidates);
            uint64_t bit = (1ULL << v);
            visitedR |= bit;
            candidates ^= bit;

            if (findAugmentingPath(matchR[v], adjMasks, matchR, visitedR, matched)) {
                matchR[v] = static_cast<uint8_t>(u);
                return true;
            }
        }
        return false;
    }
};