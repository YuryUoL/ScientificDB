#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>
#include <polyscope/curve_network.h>
#include <glm/glm.hpp>

#include "PolyscopeBridge.h"
#include "CloudAligmentEvaluator.h"

#include <imgui.h>
#include <polyscope/view.h>

namespace MatchingViz {

struct MatchRecord {
    size_t idA = 0, idB = 0;               // molecule IDs (NOT list indices)
    std::string nameA, nameB;              // polyscope cloud names
    std::vector<size_t> matching;          // matching[i] = index in B matched to atom i of A
    float bottleneck = 0.0f;
    float R[9] = {0};                      // row-major, maps centered A -> centered B
    float t[3] = {0, 0, 0};
    bool visible = true;                   // manual hide/show flag
    glm::mat4 lastA = glm::mat4(0.0f);     // transforms used for last line update
    glm::mat4 lastB = glm::mat4(0.0f);
};

inline std::vector<MatchRecord> s_matches;

inline bool  s_showAllMatches = true;                 // global master switch
inline float s_lineThickness  = 1.5f;
inline float s_lineColor[4]   = {1.0f, 0.8f, 0.1f, 0.9f};

inline std::string cloudName(size_t id, const std::string& filename) {
    return "Molecule_" + std::to_string(id) + "_" + filename;
}

inline std::string netName(const MatchRecord& m) {
    return "Match_" + std::to_string(m.idA) + "_" + std::to_string(m.idB);
}

// Finds a record for the pair in either order
inline MatchRecord* find(size_t a, size_t b) {
    for (auto& m : s_matches) {
        if ((m.idA == a && m.idB == b) || (m.idA == b && m.idB == a)) return &m;
    }
    return nullptr;
}

inline void dropNetwork(MatchRecord& m) {
    if (polyscope::state::initialized && polyscope::hasCurveNetwork(netName(m))) {
        polyscope::removeCurveNetwork(netName(m), false);
    }
    m.lastA = glm::mat4(0.0f);
    m.lastB = glm::mat4(0.0f);
}

inline bool remove(size_t a, size_t b) {
    for (size_t i = 0; i < s_matches.size(); ++i) {
        const auto& m = s_matches[i];
        if ((m.idA == a && m.idB == b) || (m.idA == b && m.idB == a)) {
            dropNetwork(s_matches[i]);
            s_matches.erase(s_matches.begin() + i);
            return true;
        }
    }
    return false;
}

// Called when a molecule is deleted from the dataset
inline void removeForMolecule(size_t id) {
    for (size_t i = s_matches.size(); i-- > 0;) {
        if (s_matches[i].idA == id || s_matches[i].idB == id) {
            dropNetwork(s_matches[i]);
            s_matches.erase(s_matches.begin() + i);
        }
    }
}

inline void clearAll() {
    for (auto& m : s_matches) dropNetwork(m);
    s_matches.clear();
}

// Stores (or overwrites) the matching for a pair
inline void store(size_t idA, size_t idB, const std::string& fileA, const std::string& fileB,
                  const AlignmentResult& res) {
    remove(idA, idB);
    MatchRecord m;
    m.idA = idA;
    m.idB = idB;
    m.nameA = cloudName(idA, fileA);
    m.nameB = cloudName(idB, fileB);
    m.matching = res.matching;
    m.bottleneck = res.bestBottleneck;
    for (int i = 0; i < 9; ++i) m.R[i] = res.rotation[i];
    for (int i = 0; i < 3; ++i) m.t[i] = res.translation[i];
    m.visible = false;   // hidden until the user runs show_match
    s_matches.push_back(std::move(m));
}

// Call once per frame from the main loop, before drawFrame().
// Lines are shown only while the record is visible AND both clouds are rendered/enabled.
// Draws matching lines in screen space. Runs inside the ImGui frame
// (same place the atom labels are drawn), so it always matches the gizmo.
inline void drawLines() {
    if (!polyscope::state::initialized || !s_showAllMatches || s_matches.empty()) return;

    const glm::mat4 view = polyscope::view::getCameraViewMatrix();
    const glm::mat4 proj = polyscope::view::getCameraPerspectiveMatrix();
    const ImVec2 display = ImGui::GetIO().DisplaySize;
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    const ImU32 col = ImGui::ColorConvertFloat4ToU32(
        ImVec4(s_lineColor[0], s_lineColor[1], s_lineColor[2], s_lineColor[3]));

    auto project = [&](const glm::mat4& mvp, const glm::vec3& p, ImVec2& out) -> bool {
        glm::vec4 clip = mvp * glm::vec4(p, 1.0f);
        if (clip.w <= 0.0f) return false;                     // behind camera
        glm::vec3 ndc = glm::vec3(clip) / clip.w;
        out = ImVec2((ndc.x * 0.5f + 0.5f) * display.x,
                     (1.0f - (ndc.y * 0.5f + 0.5f)) * display.y);
        return true;
    };

    for (const auto& m : s_matches) {
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
        const glm::mat4 mvpA = proj * view * ca->getTransform();   // includes gizmo
        const glm::mat4 mvpB = proj * view * cb->getTransform();

        for (size_t i = 0; i < m.matching.size() && i < locA.size(); ++i) {
            size_t j = m.matching[i];
            if (j >= locB.size()) continue;
            ImVec2 a, b;
            if (project(mvpA, locA[i], a) && project(mvpB, locB[j], b))
                dl->AddLine(a, b, col, s_lineThickness);
        }
    }
}

// Call once after PolyscopeBridge::init(). Chains onto the existing extra callback.
inline void install() {
    auto prev = PolyscopeBridge::s_extraUserCallback;
    PolyscopeBridge::s_extraUserCallback = [prev]() {
        drawLines();
        ImGui::Checkbox("Show matching lines", &s_showAllMatches);
        ImGui::SliderFloat("Line thickness", &s_lineThickness, 0.5f, 5.0f);
        ImGui::ColorEdit4("Line color", s_lineColor, ImGuiColorEditFlags_NoInputs);
        if (prev) prev();
    };
}

inline void setAllVisible(bool v) {
    for (auto& m : s_matches) m.visible = v;
}

// Snap cloud A onto cloud B using the stored rotation (record must be stored as A -> B)
inline bool applyAlignment(size_t idA, size_t idB) {
    for (auto& m : s_matches) {
        if (m.idA != idA || m.idB != idB) continue;
        if (!polyscope::hasPointCloud(m.nameA) || !polyscope::hasPointCloud(m.nameB)) {
            std::cout << "[Matching] Both molecules must be viewed first.\n";
            return false;
        }
        // glm is column-major, R is row-major
        glm::mat4 R4(1.0f);
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                R4[c][r] = m.R[r * 3 + c];

        glm::mat4 TB = polyscope::getPointCloud(m.nameB)->getTransform();
        polyscope::getPointCloud(m.nameA)->setTransform(TB * R4);
        return true;
    }
    std::cout << "[Matching] No stored matching for this exact order (A -> B).\n";
    return false;
}

inline void list() {
    std::cout << "\n[Matching] Stored matchings: " << s_matches.size() << "\n";
    for (const auto& m : s_matches) {
        std::cout << "  idA=" << m.idA << "  idB=" << m.idB
                  << "  bottleneck=" << std::fixed << std::setprecision(6) << m.bottleneck
                  << "  pairs=" << m.matching.size()
                  << "  " << (m.visible ? "visible" : "hidden") << "\n";
    }
    std::cout << "\n";
}

} // namespace MatchingViz