#pragma once
#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>
#include <polyscope/view.h>
#include <imgui.h>

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <iostream>
#include <cfloat>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/euler_angles.hpp>

struct AtomRecord {
    std::string symbol;
    double x, y, z;
    double charge;
};

namespace PolyscopeBridge {
    inline bool s_gizmosEnabled = true;
    inline std::string s_activeCloudName = "";

    // ---- Label settings ----
    inline bool  s_showLabels     = true;
    inline bool  s_hideHydrogens  = false;  // handy for big molecules
    inline float s_labelFontScale = 1.0f;
    inline int   s_labelIndexBase = 0;      // set to 1 if you want atoms numbered from 1

    // Optional: your own UI code can go here, it runs inside the Polyscope ImGui frame
    inline std::function<void()> s_extraUserCallback = nullptr;

    // Per-molecule label data (positions are in the cloud's local space)
    struct MoleculeLabels {
        std::vector<glm::vec3>   localPositions;
        std::vector<std::string> labels;
        std::vector<bool>        isHydrogen;
    };
    inline std::map<std::string, MoleculeLabels> s_labelData;

    // Projects every atom to screen space and draws its text label
    inline void drawAtomLabels() {
        if (!s_showLabels || s_labelData.empty()) return;

        const glm::mat4 view = polyscope::view::getCameraViewMatrix();
        const glm::mat4 proj = polyscope::view::getCameraPerspectiveMatrix();

        ImDrawList* dl = ImGui::GetBackgroundDrawList();
        const ImVec2 display = ImGui::GetIO().DisplaySize;
        const float fontSize = ImGui::GetFontSize() * s_labelFontScale;
        ImFont* font = ImGui::GetFont();

        for (const auto& [cloudName, data] : s_labelData) {
            if (!polyscope::hasPointCloud(cloudName)) continue;
            polyscope::PointCloud* cloud = polyscope::getPointCloud(cloudName);
            if (!cloud->isEnabled()) continue;

            // Includes gizmo rotation/translation
            const glm::mat4 mvp = proj * view * cloud->getTransform();

            for (size_t i = 0; i < data.labels.size(); ++i) {
                if (s_hideHydrogens && data.isHydrogen[i]) continue;

                glm::vec4 clip = mvp * glm::vec4(data.localPositions[i], 1.0f);
                if (clip.w <= 0.0f) continue;                       // behind camera
                glm::vec3 ndc = glm::vec3(clip) / clip.w;
                if (ndc.x < -1.f || ndc.x > 1.f || ndc.y < -1.f || ndc.y > 1.f) continue;

                ImVec2 screen((ndc.x * 0.5f + 0.5f) * display.x,
                              (1.0f - (ndc.y * 0.5f + 0.5f)) * display.y);

                const char* text = data.labels[i].c_str();
                ImVec2 size = font->CalcTextSizeA(fontSize, FLT_MAX, 0.0f, text);
                ImVec2 pos(screen.x - size.x * 0.5f, screen.y - size.y - 6.0f); // just above the atom

                // Black shadow + white text so it's readable on any background
                dl->AddText(font, fontSize, ImVec2(pos.x + 1, pos.y + 1), IM_COL32(0, 0, 0, 255), text);
                dl->AddText(font, fontSize, pos, IM_COL32(255, 255, 255, 255), text);
            }
        }
    }

    inline void init() {
        polyscope::init();
        polyscope::options::autocenterStructures = true;

        polyscope::state::userCallback = []() {
            ImGui::Checkbox("Show atom labels", &s_showLabels);
            ImGui::Checkbox("Hide hydrogen labels", &s_hideHydrogens);
            ImGui::SliderFloat("Label size", &s_labelFontScale, 0.5f, 2.5f);

            drawAtomLabels();

            if (s_extraUserCallback) s_extraUserCallback();
        };
    }

    inline void drawFrame() {
        polyscope::frameTick();
    }

    inline void visualizeMolecule(size_t id, const std::string& name, const std::vector<AtomRecord>& atoms) {
        if (atoms.empty()) return;

        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> colors;
        MoleculeLabels labelData;
        positions.reserve(atoms.size());
        colors.reserve(atoms.size());
        labelData.labels.reserve(atoms.size());
        labelData.isHydrogen.reserve(atoms.size());

        // 1. Geometric center
        glm::vec3 center(0.0f);
        for (const auto& atom : atoms) {
            center += glm::vec3(atom.x, atom.y, atom.z);
        }
        center /= static_cast<float>(atoms.size());

        // 2. Positions centered at (0,0,0) so the gizmo pivots around the molecule center
        for (size_t i = 0; i < atoms.size(); ++i) {
            const auto& atom = atoms[i];
            glm::vec3 shiftedPos = glm::vec3(atom.x, atom.y, atom.z) - center;
            positions.push_back(shiftedPos);

            glm::vec3 color(0.5f, 0.5f, 0.5f); // Default gray
            if (atom.symbol == "C") {
                color = glm::vec3(0.18f, 0.18f, 0.18f);
            } else if (atom.symbol == "H") {
                color = glm::vec3(0.85f, 0.95f, 1.0f);
            } else if (atom.symbol == "O") {
                color = glm::vec3(0.9f, 0.12f, 0.12f);
            } else if (atom.symbol == "N") {
                color = glm::vec3(0.15f, 0.4f, 0.9f);
            } else if (atom.symbol == "F" || atom.symbol == "Cl") {
                color = glm::vec3(0.12f, 0.85f, 0.25f);
            }
            colors.push_back(color);

            // Label = symbol + index, e.g. "C13"
            labelData.labels.push_back(atom.symbol + std::to_string(i + s_labelIndexBase));
            labelData.isHydrogen.push_back(atom.symbol == "H");
        }
        labelData.localPositions = positions;

        s_activeCloudName = "Molecule_" + std::to_string(id) + "_" + name;
        polyscope::PointCloud* psCloud = polyscope::registerPointCloud(s_activeCloudName, positions);

        auto* colorQ = psCloud->addColorQuantity("Element Colors", colors);
        colorQ->setEnabled(true);

        psCloud->setPointRadius(0.08f, true);
        psCloud->setTransformGizmoEnabled(s_gizmosEnabled);

        // 3. Move the cloud back to its original world position
        glm::mat4 initialTransform = glm::translate(glm::mat4(1.0f), center);
        psCloud->setTransform(initialTransform);

        // Store labels for this cloud
        s_labelData[s_activeCloudName] = std::move(labelData);
    }

    // Removes a rendered molecule (for 'unview')
    inline void removeVisualization(size_t id, const std::string& name) {
        std::string cloudName = "Molecule_" + std::to_string(id) + "_" + name;
        if (polyscope::hasPointCloud(cloudName)) {
            polyscope::removePointCloud(cloudName, true);
            s_labelData.erase(cloudName);
            std::cout << "[Polyscope] Unviewed / removed structure: " << cloudName << "\n";
            if (s_activeCloudName == cloudName) {
                s_activeCloudName = "";
            }
        } else {
            std::cout << "[Polyscope] Warning: Structure '" << cloudName << "' is not currently rendered.\n";
        }
    }

    inline void setGizmoEnabled(bool enabled) {
        s_gizmosEnabled = enabled;
        if (!s_activeCloudName.empty()) {
            if (auto* psCloud = polyscope::getPointCloud(s_activeCloudName)) {
                psCloud->setTransformGizmoEnabled(enabled);
            }
        }
    }

    inline void translateActiveCloud(float x, float y, float z) {
        if (!s_activeCloudName.empty()) {
            if (auto* psCloud = polyscope::getPointCloud(s_activeCloudName)) {
                glm::mat4 currentT = psCloud->getTransform();
                glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
                psCloud->setTransform(translation * currentT);
            }
        }
    }

    inline void transformActiveCloud(glm::vec3 rotDeg, glm::vec3 trans) {
        if (!s_activeCloudName.empty()) {
            if (auto* psCloud = polyscope::getPointCloud(s_activeCloudName)) {
                glm::mat4 mat(1.0f);
                mat = glm::translate(mat, trans);
                mat = glm::rotate(mat, glm::radians(rotDeg.z), glm::vec3(0.0f, 0.0f, 1.0f));
                mat = glm::rotate(mat, glm::radians(rotDeg.y), glm::vec3(0.0f, 1.0f, 0.0f));
                mat = glm::rotate(mat, glm::radians(rotDeg.x), glm::vec3(1.0f, 0.0f, 0.0f));
                psCloud->setTransform(mat);
            }
        }
    }
}