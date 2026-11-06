#pragma once
#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>
#include <vector>
#include <string>
#include <iostream>
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

    inline void init() {
        polyscope::init();
        polyscope::options::autocenterStructures = true;
    }

    inline void drawFrame() {
        polyscope::frameTick();
    }

    inline void visualizeMolecule(size_t id, const std::string& name, const std::vector<AtomRecord>& atoms) {
        if (atoms.empty()) return;

        std::vector<glm::vec3> positions;
        std::vector<glm::vec3> colors;
        positions.reserve(atoms.size());
        colors.reserve(atoms.size());

        // 1. Compute the geometric center (or center of mass)
        glm::vec3 center(0.0f);
        for (const auto& atom : atoms) {
            center += glm::vec3(atom.x, atom.y, atom.z);
        }
        center /= static_cast<float>(atoms.size());

        // 2. Populate positions centered at (0,0,0) so the rotation gizmo pivots around the molecule center
        for (const auto& atom : atoms) {
            glm::vec3 shiftedPos = glm::vec3(atom.x, atom.y, atom.z) - center;
            positions.push_back(shiftedPos);

            // Standard chemical element coloring schema with a distinct, vivid Hydrogen tone
            glm::vec3 color(0.5f, 0.5f, 0.5f); // Default gray
            if (atom.symbol == "C") {
                color = glm::vec3(0.18f, 0.18f, 0.18f); // Carbon: Dark Charcoal
            } else if (atom.symbol == "H") {
                color = glm::vec3(0.85f, 0.95f, 1.0f);   // Hydrogen: Bright Ice Blue / Clean White-Cyan
            } else if (atom.symbol == "O") {
                color = glm::vec3(0.9f, 0.12f, 0.12f);   // Oxygen: Vivid Red
            } else if (atom.symbol == "N") {
                color = glm::vec3(0.15f, 0.4f, 0.9f);    // Nitrogen: Deep Royal Blue
            } else if (atom.symbol == "F" || atom.symbol == "Cl") {
                color = glm::vec3(0.12f, 0.85f, 0.25f);  // Halogens: Emerald Green
            }
            colors.push_back(color);
        }

        s_activeCloudName = "Molecule_" + std::to_string(id) + "_" + name;
        polyscope::PointCloud* psCloud = polyscope::registerPointCloud(s_activeCloudName, positions);
        
        auto* colorQ = psCloud->addColorQuantity("Element Colors", colors);
        colorQ->setEnabled(true);
        
        psCloud->setPointRadius(0.08f, true);
        psCloud->setTransformGizmoEnabled(s_gizmosEnabled);

        // 3. Offset the cloud back to its original world position via transformation matrix 
        //    while keeping the gizmo origin anchored right at the center of mass.
        glm::mat4 initialTransform = glm::translate(glm::mat4(1.0f), center);
        psCloud->setTransform(initialTransform);
    }

    // Removes a specific rendered molecule point cloud from Polyscope (for 'unview')
    inline void removeVisualization(size_t id, const std::string& name) {
        std::string cloudName = "Molecule_" + std::to_string(id) + "_" + name;
        if (polyscope::hasPointCloud(cloudName)) {
            polyscope::removePointCloud(cloudName, true);
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

    // Direct translation matrix setter for the active molecular point cloud
    inline void translateActiveCloud(float x, float y, float z) {
        if (!s_activeCloudName.empty()) {
            if (auto* psCloud = polyscope::getPointCloud(s_activeCloudName)) {
                glm::mat4 currentT = psCloud->getTransform();
                glm::mat4 translation = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z));
                psCloud->setTransform(translation * currentT);
            }
        }
    }

    // Direct rotation & translation setter using Euler angles (degrees)
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