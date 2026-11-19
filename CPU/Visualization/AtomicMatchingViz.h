#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>
#include <glm/glm.hpp>

#include "PolyscopeBridge.h"
#include "CPU/Aligners/AtomicCloudAlignmentEvaluator.h"
#include <imgui.h>
#include <polyscope/view.h>

namespace AtomicMatchingViz {

struct AtomicMatchRecord {
    size_t idA = 0, idB = 0;               // Atomic cloud indices
    std::string nameA, nameB;              // Polyscope cloud names
    std::vector<size_t> matching;          // matching[i] = index in full B matched to subset A[i]
    std::vector<size_t> subsetIndicesA;    // The actual atom indices from cloud A that were aligned
    std::vector<size_t> subsetIndicesB;    // The actual atom indices from cloud B that were aligned
    float bottleneck = 0.0f;
    float R[9] = {0};                      // rotation matrix
    bool visible = false;
};

inline std::vector<AtomicMatchRecord> s_atomicMatches;
inline bool  s_showAllAtomicMatches = true;
inline float s_atomicLineThickness  = 1.5f;
inline float s_atomicLineColor[4]   = {0.1f, 0.9f, 0.8f, 0.9f};

inline std::string atomicCloudName(size_t atomicIdx, size_t molIdx,
                                   const std::string& symbol, size_t atomIdx) {
    return "AtomicCloud_" + std::to_string(atomicIdx) + "_" +
           std::to_string(molIdx) + "_" + symbol + std::to_string(atomIdx);
}

inline AtomicMatchRecord* find(size_t a, size_t b) {
    for (auto& m : s_atomicMatches) {
        if (m.idA == a && m.idB == b) return &m;
    }
    return nullptr;
}

inline bool remove(size_t a, size_t b) {
    for (size_t i = 0; i < s_atomicMatches.size(); ++i) {
        if (s_atomicMatches[i].idA == a && s_atomicMatches[i].idB == b) {
            s_atomicMatches.erase(s_atomicMatches.begin() + i);
            return true;
        }
    }
    return false;
}

inline void clearAll() {
    s_atomicMatches.clear();
}

inline void store(size_t idA, size_t idB,
                  size_t molIdxA, const std::string& symA, size_t atomIdxA,
                  size_t molIdxB, const std::string& symB, size_t atomIdxB,
                  const SubsetAlignmentResult& res,
                  const std::vector<size_t>& subA, const std::vector<size_t>& subB) {
    remove(idA, idB);
    AtomicMatchRecord m;
    m.idA = idA;
    m.idB = idB;
    m.nameA = atomicCloudName(idA, molIdxA, symA, atomIdxA);
    m.nameB = atomicCloudName(idB, molIdxB, symB, atomIdxB);
    m.matching = res.matching;
    m.subsetIndicesA = subA;
    m.subsetIndicesB = subB;
    m.bottleneck = res.bestBottleneck;
    for (int i = 0; i < 9; ++i) m.R[i] = res.rotation[i];
    m.visible = false;
    s_atomicMatches.push_back(std::move(m));
}

inline void drawLines() {
    if (!polyscope::state::initialized || !s_showAllAtomicMatches || s_atomicMatches.empty()) return;

    const glm::mat4 view = polyscope::view::getCameraViewMatrix();
    const glm::mat4 proj = polyscope::view::getCameraPerspectiveMatrix();
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const ImU32 col = ImGui::ColorConvertFloat4ToU32(
        ImVec4(s_atomicLineColor[0], s_atomicLineColor[1], s_atomicLineColor[2], s_atomicLineColor[3]));

    auto project = [&](const glm::mat4& mvp, const glm::vec3& p, ImVec2& out) -> bool {
        glm::vec4 clip = mvp * glm::vec4(p, 1.0f);
        if (clip.w <= 0.0f) return false;
        glm::vec3 ndc = glm::vec3(clip) / clip.w;
        out = ImVec2((ndc.x * 0.5f + 0.5f) * display.x,
                     (1.0f - (ndc.y * 0.5f + 0.5f)) * display.y);
        return true;
    };

    for (const auto& m : s_atomicMatches) {
        if (!m.visible || m.matching.empty()) continue;
        if (!polyscope::hasPointCloud(m.nameA) || !polyscope::hasPointCloud(m.nameB)) continue;

        auto* ca = polyscope::getPointCloud(m.nameA);
        auto* cb = polyscope::getPointCloud(m.nameB);
        if (!ca->isEnabled() || !cb->isEnabled()) continue;

        auto itA = PolyscopeBridge::s_labelData.find(m.nameA);
        auto itB = PolyscopeBridge::s_labelData.find(m.nameB);
        if (itA == PolyscopeBridge::s_labelData.end() || itB == PolyscopeBridge::s_labelData.end()) continue;

        const auto& locA = itA->second.localPositions;
        const auto& locB = itB->second.localPositions;
        const glm::mat4 mvpA = proj * view * ca->getTransform();
        const glm::mat4 mvpB = proj * view * cb->getTransform();

        for (size_t i = 0; i < m.subsetIndicesA.size() && i < m.matching.size(); ++i) {
            size_t atomIdxA = m.subsetIndicesA[i];
            size_t atomIdxB = m.matching[i];

            if (atomIdxA >= locA.size() || atomIdxB >= locB.size()) continue;

            ImVec2 a, b;
            if (project(mvpA, locA[atomIdxA], a) && project(mvpB, locB[atomIdxB], b))
                dl->AddLine(a, b, col, s_atomicLineThickness);
        }
    }
}

inline void install() {
    auto prev = PolyscopeBridge::s_extraUserCallback;
    PolyscopeBridge::s_extraUserCallback = [prev]() {
        drawLines();
        ImGui::Checkbox("Show atomic matching lines", &s_showAllAtomicMatches);
        ImGui::SliderFloat("Atomic line thickness", &s_atomicLineThickness, 0.5f, 5.0f);
        ImGui::ColorEdit4("Atomic line color", s_atomicLineColor, ImGuiColorEditFlags_NoInputs);
        if (prev) prev();
    };
}

inline bool applyAlignment(size_t idA, size_t idB) {
    for (auto& m : s_atomicMatches) {
        if (m.idA != idA || m.idB != idB) continue;
        if (!polyscope::hasPointCloud(m.nameA) || !polyscope::hasPointCloud(m.nameB)) {
            std::cout << "[AtomicMatching] Both atomic clouds must be viewed first.\n";
            return false;
        }
        glm::mat4 R4(1.0f);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R4[c][r] = m.R[r * 3 + c];

        glm::mat4 TB = polyscope::getPointCloud(m.nameB)->getTransform();
        polyscope::getPointCloud(m.nameA)->setTransform(TB * R4);
        return true;
    }
    std::cout << "[AtomicMatching] No stored atomic matching for this pair.\n";
    return false;
}

inline void setVisible(size_t idA, size_t idB, bool v) {
    if (auto* m = find(idA, idB)) m->visible = v;
}

} // namespace AtomicMatchingViz