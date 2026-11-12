#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <imgui.h>
#include "PolyscopeBridge.h"
#include "Duplicatechecker.h"

class MolecularGraphViz
{
public:
    struct ClusterInfo
    {
        int clusterId;
        size_t atomCount;
        std::vector<size_t> memberIndices;
        std::vector<DupMatch> internalMatches;
    };

    inline static bool s_showClusterWindow = false;
    inline static std::vector<ClusterInfo> s_cachedClusters;
    inline static int s_activeVisualizedCluster = -1;

    static void setGraphVisible(bool visible)
    {
        try
        {
            if (polyscope::hasPointCloud("Duplicate Component"))
            {
                polyscope::getPointCloud("Duplicate Component")->setEnabled(visible);
            }
            if (polyscope::hasCurveNetwork("Component Edges"))
            {
                polyscope::getCurveNetwork("Component Edges")->setEnabled(visible);
            }
        }
        catch (...)
        {
            std::cout << "[GraphViz] Warning: Component structures not initialized.\n";
        }
    }

    static void clearGraph()
    {
        try
        {
            if (polyscope::hasPointCloud("Duplicate Component"))
            {
                polyscope::removePointCloud("Duplicate Component", true);
            }
            if (polyscope::hasCurveNetwork("Component Edges"))
            {
                polyscope::removeCurveNetwork("Component Edges", true);
            }
            PolyscopeBridge::s_labelData.erase("Duplicate Component");
            s_activeVisualizedCluster = -1;
            std::cout << "[GraphViz] Component safely cleared from Polyscope.\n";
        }
        catch (...)
        {
            std::cout << "[GraphViz] Warning: Failed to clear structures.\n";
        }
    }

    static void buildClusters(const std::vector<DupMatch>& matches, 
                              const std::vector<size_t>& atomCounts)
    {
        s_cachedClusters.clear();
        if (matches.empty())
        {
            std::cout << "[GraphViz] Error: No matches provided to build clusters.\n";
            return;
        }

        std::unordered_set<size_t> activeNodes;
        std::unordered_map<size_t, std::vector<size_t>> adj;

        for (const auto& m : matches)
        {
            activeNodes.insert(m.queryIdx);
            activeNodes.insert(m.matchIdx);
            adj[m.queryIdx].push_back(m.matchIdx);
            adj[m.matchIdx].push_back(m.queryIdx);
        }

        std::unordered_map<size_t, int> nodeCluster;
        std::unordered_set<size_t> visited;
        int cId = 0;

        for (size_t startNode : activeNodes)
        {
            if (visited.count(startNode)) continue;

            std::queue<size_t> q;
            q.push(startNode);
            visited.insert(startNode);

            ClusterInfo cluster;
            cluster.clusterId = cId;

            while (!q.empty())
            {
                size_t u = q.front();
                q.pop();
                cluster.memberIndices.push_back(u);
                cluster.atomCount = atomCounts[u];
                nodeCluster[u] = cId;

                for (size_t v : adj[u])
                {
                    if (!visited.count(v))
                    {
                        visited.insert(v);
                        q.push(v);
                    }
                }
            }

            for (const auto& m : matches)
            {
                if (nodeCluster.count(m.queryIdx) && nodeCluster[m.queryIdx] == cId &&
                    nodeCluster.count(m.matchIdx) && nodeCluster[m.matchIdx] == cId)
                {
                    cluster.internalMatches.push_back(m);
                }
            }

            s_cachedClusters.push_back(cluster);
            cId++;
        }

        s_showClusterWindow = true;
        std::cout << "[GraphViz] Computed " << s_cachedClusters.size() << " connected components. Opening Explorer Window.\n";
    }

    static void visualizeCluster(int clusterId, const std::vector<std::string>& filenames, 
                                 const std::vector<size_t>& atomCounts)
    {
        if (clusterId < 0 || clusterId >= static_cast<int>(s_cachedClusters.size())) return;
        
        clearGraph();
        const auto& cluster = s_cachedClusters[clusterId];
        s_activeVisualizedCluster = clusterId;

        const auto& nodes = cluster.memberIndices;
        size_t nNodes = nodes.size();

        std::vector<glm::vec3> points;
        points.reserve(nNodes);
        std::unordered_map<size_t, size_t> nodeToIdxMap;
        PolyscopeBridge::MoleculeLabels labelData;
        labelData.labels.reserve(nNodes);
        labelData.isHydrogen.reserve(nNodes);

        float radius = std::min(3.0f + 1.0f * nNodes, 15.0f);
        for (size_t i = 0; i < nNodes; ++i)
        {
            size_t u = nodes[i];
            float theta = static_cast<float>(i) * 2.0f * M_PI / nNodes;
            float x = radius * std::cos(theta);
            float y = (i % 2 == 0 ? 1.0f : -1.0f) * 1.5f;
            float z = radius * std::sin(theta);

            points.push_back(glm::vec3(x, y, z));
            nodeToIdxMap[u] = i;

            labelData.labels.push_back("ID: " + std::to_string(u));
            labelData.isHydrogen.push_back(false);
        }
        labelData.localPositions = points;

        std::vector<glm::uvec2> edges;
        for (const auto& m : cluster.internalMatches)
        {
            if (nodeToIdxMap.count(m.queryIdx) && nodeToIdxMap.count(m.matchIdx))
            {
                edges.push_back(glm::uvec2(nodeToIdxMap[m.queryIdx], nodeToIdxMap[m.matchIdx]));
            }
        }

        auto* psCloud = polyscope::registerPointCloud("Duplicate Component", points);
        std::vector<glm::vec3> colors(nNodes, glm::vec3(0.15f, 0.65f, 0.95f));
        auto* colorQ = psCloud->addColorQuantity("Uniform Color", colors);
        colorQ->setEnabled(true);
        psCloud->setPointRadius(0.04f, true);

        PolyscopeBridge::s_labelData["Duplicate Component"] = std::move(labelData);

        if (!edges.empty())
        {
            auto* psNetwork = polyscope::registerCurveNetwork("Component Edges", points, edges);
            psNetwork->setColor(glm::vec3(0.85f, 0.35f, 0.15f));
        }

        std::cout << "[GraphViz] Visualizing Component " << clusterId << " (" << nNodes << " nodes, " << edges.size() << " edges).\n";
    }

    static void drawImGuiWindow(const std::vector<std::string>& filenames, const std::vector<size_t>& atomCounts)
    {
        if (!s_showClusterWindow) return;

        ImGui::SetNextWindowSize(ImVec2(600, 450), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Duplicate Component Explorer", &s_showClusterWindow))
        {
            ImGui::Text("Total Connected Components: %zu", s_cachedClusters.size());
            if (s_activeVisualizedCluster >= 0)
            {
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(0.2f, 0.8f, 0.2f, 1.0f), "[Currently Viewing: Component %d]", s_activeVisualizedCluster);
            }
            ImGui::Separator();

            if (ImGui::BeginTable("ClusterTable", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY))
            {
                ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_WidthFixed, 50.0f);
                ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, 65.0f);
                ImGui::TableSetupColumn("Molecule Indices", ImGuiTableColumnFlags_WidthStretch);
                ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_WidthFixed, 95.0f);
                ImGui::TableHeadersRow();

                for (const auto& cluster : s_cachedClusters)
                {
                    ImGui::TableNextRow();

                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("%d", cluster.clusterId);

                    ImGui::TableSetColumnIndex(1);
                    ImGui::Text("%zu nodes", cluster.memberIndices.size());

                    ImGui::TableSetColumnIndex(2);
                    std::string idList = "";
                    for (size_t u : cluster.memberIndices) idList += std::to_string(u) + " ";
                    ImGui::Text("%s", idList.c_str());

                    ImGui::TableSetColumnIndex(3);
                    std::string btnName = "Visualize##" + std::to_string(cluster.clusterId);
                    if (ImGui::Button(btnName.c_str()))
                    {
                        visualizeCluster(cluster.clusterId, filenames, atomCounts);
                    }
                }
                ImGui::EndTable();
            }
        }
        ImGui::End();
    }
};