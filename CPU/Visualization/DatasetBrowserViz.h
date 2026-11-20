#pragma once
#include <imgui.h>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include "CPU/MolecularDatasetStore.h"
#include "CPU/MolecularOperations.h"
#include "PolyscopeBridge.h"

namespace DatasetBrowserViz {
    inline bool s_windowVisible = false;

    inline std::string computeCompositionSummary(const MolecularDatasetStore& store, size_t idx) {
        std::string rawContent = store.getRawContent(idx);
        std::map<std::string, int> elementCounts;
        std::istringstream iss(rawContent);
        std::string line;
        std::getline(iss, line); // Skip atom count line
        std::getline(iss, line); // Skip comment line
        while (std::getline(iss, line)) {
            std::istringstream atomStream(line);
            std::string sym;
            if (atomStream >> sym) {
                elementCounts[sym]++;
            }
        }
        std::string compStr;
        for (const auto& [sym, count] : elementCounts) {
            compStr += sym + ":" + std::to_string(count) + " ";
        }
        return compStr;
    }

    inline void drawWindow(MolecularDatasetStore& store) {
        if (!s_windowVisible) return;

        // Cache composition strings lazily only for rows that actually scroll into view
        static std::vector<std::string> s_compCache;
        static size_t s_cachedStoreSize = 0;
        const size_t totalMols = store.size();

        if (s_cachedStoreSize != totalMols) {
            s_compCache.assign(totalMols, std::string());
            s_cachedStoreSize = totalMols;
        }

        ImGui::SetNextWindowSize(ImVec2(650, 450), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Molecular Dataset Browser", &s_windowVisible)) {
            ImGui::Text("Total Loaded Molecules: %zu", totalMols);
            ImGui::Separator();

            ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                    ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable;

            if (ImGui::BeginTable("MoleculeBrowserTable", 5, flags, ImVec2(0, 350))) {
                ImGui::TableSetupScrollFreeze(0, 1); // Keep header visible while scrolling
                ImGui::TableSetupColumn("Idx", ImGuiTableColumnFlags_WidthFixed, 55.0f);
                ImGui::TableSetupColumn("Filename", ImGuiTableColumnFlags_WidthStretch, 150.0f);
                ImGui::TableSetupColumn("Atoms", ImGuiTableColumnFlags_WidthFixed, 50.0f);
                ImGui::TableSetupColumn("Composition", ImGuiTableColumnFlags_WidthStretch, 120.0f);
                ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, 100.0f);
                ImGui::TableHeadersRow();

                ImGuiListClipper clipper;
                clipper.Begin(static_cast<int>(totalMols));
                MolecularOperations molOps(store);
                while (clipper.Step()) {
                    for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                        const size_t i = static_cast<size_t>(row);
                        ImGui::TableNextRow();

                        // 1. Index
                        ImGui::TableSetColumnIndex(0);
                        ImGui::Text("%zu", i);

                        // 2. Filename
                        ImGui::TableSetColumnIndex(1);
                        std::string filename = store.getFilename(i);
                        ImGui::TextUnformatted(filename.c_str());

                        // 3. Atom Count
                        ImGui::TableSetColumnIndex(2);
                        ImGui::Text("%zu", store.getAtomCount(i));

                        // 4. Composition Summary (Computed once per visible row, then cached)
                        ImGui::TableSetColumnIndex(3);
                        if (s_compCache[i].empty()) {
                            s_compCache[i] = computeCompositionSummary(store, i);
                        }
                        ImGui::TextUnformatted(s_compCache[i].c_str());

                        // 5. Action Buttons (View / Hide)
                        ImGui::TableSetColumnIndex(4);
                        ImGui::PushID(row);
                        if (ImGui::Button("View")) {
                            molOps.visualizeIndex(i);
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Hide")) {
                            molOps.unviewIndex(i);
                        }
                        ImGui::PopID();
                    }
                }
                ImGui::EndTable();
            }
        }
        ImGui::End();
    }
}