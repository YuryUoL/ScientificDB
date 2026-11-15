#pragma once
#include <imgui.h>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include "MolecularDatasetStore.h"
#include "PolyscopeBridge.h"

namespace DatasetBrowserViz {
    inline bool s_windowVisible = true;

    inline void drawWindow(MolecularDatasetStore& store) {
        if (!s_windowVisible) return;

        ImGui::SetNextWindowSize(ImVec2(650, 450), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Molecular Dataset Browser", &s_windowVisible)) {
            ImGui::Text("Total Loaded Molecules: %zu", store.size());
            ImGui::Separator();

            // Configure scrollable, resizable table with borders
            ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | 
                                    ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable;

            if (ImGui::BeginTable("MoleculeBrowserTable", 5, flags, ImVec2(0, 350))) {
                ImGui::TableSetupColumn("Idx", ImGuiTableColumnFlags_WidthFixed, 40.0f);
                ImGui::TableSetupColumn("Filename", ImGuiTableColumnFlags_WidthStretch, 150.0f);
                ImGui::TableSetupColumn("Atoms", ImGuiTableColumnFlags_WidthFixed, 50.0f);
                ImGui::TableSetupColumn("Composition", ImGuiTableColumnFlags_WidthStretch, 120.0f);
                ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, 100.0f);
                ImGui::TableHeadersRow();

                size_t totalMols = store.size();
                for (size_t i = 0; i < totalMols; ++i) {
                    ImGui::TableNextRow();
                    
                    // 1. Index
                    ImGui::TableSetColumnIndex(0);
                    ImGui::Text("%zu", i);

                    // 2. Filename
                    ImGui::TableSetColumnIndex(1);
                    std::string filename = store.getFilename(i);
                    ImGui::Text("%s", filename.c_str());

                    // 3. Atom Count
                    ImGui::TableSetColumnIndex(2);
                    size_t atomCount = store.getAtomCount(i);
                    ImGui::Text("%zu", atomCount);

                    // 4. Composition Summary (Counts C, H, O, N, F, etc.)
                    ImGui::TableSetColumnIndex(3);
                    std::string rawContent = store.getRawContent(i);
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
                    ImGui::Text("%s", compStr.c_str());

                    // 5. Action Buttons (View / Hide)
                    ImGui::TableSetColumnIndex(4);
                    ImGui::PushID(static_cast<int>(i));
                    if (ImGui::Button("View")) {
                        store.visualizeIndex(i);
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Hide")) {
                        store.unviewIndex(i);
                    }
                    ImGui::PopID();
                }
                ImGui::EndTable();
            }
        }
        ImGui::End();
    }
}
