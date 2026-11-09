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
                row[j] = dx * dx + dy * dy + dz * dz;
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
                for (size_t k = 0; k < N; ++k) {
                    float diff = std::abs(std::sqrt(rowA[k]) - std::sqrt(rowB[k]));
                    if (diff > maxDiff) maxDiff = diff;
                }
                costMatrixOut[i * N + j] = maxDiff;
            }
        }
    }

    /**
     * @brief Computes exact Bottleneck Distance accelerated via lowBound and highBound pruning.
     */
    static float computeFromCostMatrix(const float* costMatrix, size_t N, 
                                       float lowBound = 0.0f, 
                                       float highBound = std::numeric_limits<float>::max()) {
        if (N == 0) return 0.0f;
        if (N > MAX_VERTICES) {
            throw std::invalid_argument("BottleneckEvaluator: N exceeds MAX_VERTICES (64)");
        }

        if (lowBound >= highBound) return highBound;

        float thresholds[MAX_VERTICES * MAX_VERTICES];
        size_t numThresholds = N * N;

        for (size_t i = 0; i < numThresholds; ++i) {
            thresholds[i] = costMatrix[i];
        }
        std::sort(thresholds, thresholds + numThresholds);
        
        size_t uniqueCount = 0;
        for (size_t i = 0; i < numThresholds; ++i) {
            if (i == 0 || thresholds[i] != thresholds[i - 1]) {
                thresholds[uniqueCount++] = thresholds[i];
            }
        }

        // Restrict binary search range to [lowBound, highBound]
        int searchLow = 0;
        int searchHigh = static_cast<int>(uniqueCount) - 1;

        while (searchLow <= searchHigh && thresholds[searchLow] < lowBound) searchLow++;
        while (searchHigh >= searchLow && thresholds[searchHigh] > highBound) searchHigh--;

        if (searchLow > searchHigh) return highBound;

        int bestValidIdx = -1;
        int l = searchLow, h = searchHigh;

        while (l <= h) {
            int mid = l + (h - l) / 2;
            if (hasPerfectMatching(costMatrix, N, thresholds[mid])) {
                bestValidIdx = mid;
                h = mid - 1; // Try tighter bottleneck
            } else {
                l = mid + 1; // Relax bottleneck
            }
        }

        return (bestValidIdx != -1) ? thresholds[bestValidIdx] : highBound;
    }

    static bool hasPerfectMatching(const float* costMatrix, size_t N, float threshold) {
        uint64_t adjMasks[MAX_VERTICES];
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

        uint8_t matchR[MAX_VERTICES];
        for (size_t i = 0; i < N; ++i) matchR[i] = 255;

        for (size_t i = 0; i < N; ++i) {
            uint64_t visitedR = 0;
            if (!findAugmentingPath(i, adjMasks, matchR, visitedR)) {
                return false;
            }
        }

        return true;
    }

private:
    inline static bool findAugmentingPath(size_t u, const uint64_t* adjMasks, uint8_t* matchR, uint64_t& visitedR) {
        uint64_t candidates = adjMasks[u] & ~visitedR;
        while (candidates) {
            int v = __builtin_ctzll(candidates);
            uint64_t bit = (1ULL << v);
            visitedR |= bit;
            candidates ^= bit;

            if (matchR[v] == 255 || findAugmentingPath(matchR[v], adjMasks, matchR, visitedR)) {
                matchR[v] = static_cast<uint8_t>(u);
                return true;
            }
        }
        return false;
    }
};