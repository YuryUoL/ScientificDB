#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <limits>
#include "BottleneckEvaluator.h"   // only for MAX_VERTICES

class AtomicBottleneckEvaluator {
public:
    static constexpr size_t MAX_VERTICES = BottleneckEvaluator::MAX_VERTICES;

    /// Left-saturating matching: rows 0..m-1 each get a distinct column.
    /// matchROut (optional) must hold MAX_VERTICES entries; 255 = unmatched column.
    static bool matchInjective(const uint64_t* adj, size_t m, uint8_t* matchROut = nullptr) {
        uint8_t local[MAX_VERTICES];
        uint8_t* matchR = matchROut ? matchROut : local;
        for (size_t i = 0; i < MAX_VERTICES; ++i) matchR[i] = 255;
        uint64_t matched = 0;
        for (size_t i = 0; i < m; ++i) {
            if (adj[i] == 0) return false;
            uint64_t visited = 0;
            if (!augment(i, adj, matchR, visited, matched)) return false;
        }
        return true;
    }

    /// Exact bottleneck of an injective m x n assignment (row stride n), searched in [lo, hi].
    static float bottleneckInjective(const float* cost, size_t m, size_t n, float lo, float hi) {
        if (m == 0) return 0.0f;
        if (lo >= hi) return hi;

        float th[MAX_VERTICES * MAX_VERTICES];
        size_t cnt = 0;
        for (size_t i = 0; i < m * n; ++i)
            if (cost[i] >= lo && cost[i] <= hi) th[cnt++] = cost[i];
        if (cnt == 0) return hi;
        std::sort(th, th + cnt);
        cnt = std::unique(th, th + cnt) - th;

        int best = -1, l = 0, h = static_cast<int>(cnt) - 1;
        while (l <= h) {
            int mid = l + (h - l) / 2;
            if (feasible(cost, m, n, th[mid])) { best = mid; h = mid - 1; }
            else l = mid + 1;
        }
        return best >= 0 ? th[best] : hi;
    }

    /// Min over injective maps of max |a_i - b_f(i)|; a (m) and b (n) ascending-sorted, m <= n.
    static float injectiveBottleneck1D(const float* a, size_t m, const float* b, size_t n) {
        const float INF = std::numeric_limits<float>::max();
        if (m == 0) return 0.0f;
        if (m > n) return INF;
        float prev[MAX_VERTICES + 1], cur[MAX_VERTICES + 1];
        for (size_t l = 0; l <= n; ++l) prev[l] = 0.0f;
        for (size_t k = 1; k <= m; ++k) {
            cur[0] = INF;
            for (size_t l = 1; l <= n; ++l) {
                const float d = std::abs(a[k - 1] - b[l - 1]);
                cur[l] = std::min(cur[l - 1], std::max(prev[l - 1], d));
            }
            for (size_t l = 0; l <= n; ++l) prev[l] = cur[l];
        }
        return prev[n];
    }

private:
    static bool feasible(const float* cost, size_t m, size_t n, float t) {
        uint64_t adj[MAX_VERTICES];
        for (size_t i = 0; i < m; ++i) {
            uint64_t mask = 0;
            for (size_t j = 0; j < n; ++j)
                if (cost[i * n + j] <= t) mask |= (1ULL << j);
            if (mask == 0) return false;
            adj[i] = mask;
        }
        return matchInjective(adj, m, nullptr);
    }

    static bool augment(size_t u, const uint64_t* adj, uint8_t* matchR,
                        uint64_t& visited, uint64_t& matched) {
        uint64_t freeC = adj[u] & ~matched;
        if (freeC) {
            int v = __builtin_ctzll(freeC);
            matchR[v] = static_cast<uint8_t>(u);
            matched |= (1ULL << v);
            return true;
        }
        uint64_t c;
        while ((c = adj[u] & ~visited) != 0) {
            int v = __builtin_ctzll(c);
            visited |= (1ULL << v);
            if (augment(matchR[v], adj, matchR, visited, matched)) {
                matchR[v] = static_cast<uint8_t>(u);
                return true;
            }
        }
        return false;
    }
};