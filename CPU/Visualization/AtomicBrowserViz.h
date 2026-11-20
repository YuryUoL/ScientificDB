#pragma once
#include <imgui.h>
#include <string>

// Forward declare first so the function signature compiles
class MolecularDatasetStore;

namespace AtomicBrowserViz {
    inline bool s_windowVisible = false;
    void drawWindow(MolecularDatasetStore& store);
}

// Include the full store definition AFTER the function signature
#include "CPU/MolecularDatasetStore.h"
#include "CPU/AtomicOperations.h"


namespace AtomicBrowserViz {
    inline void drawWindow(MolecularDatasetStore& store) {
        if (!s_windowVisible) return;

        AtomicOperations atomOps(store);

        const size_t totalAtomic = atomOps.atomicSize();

        ImGui::SetNextWindowSize(ImVec2(650, 450), ImGuiCond_FirstUseEver);
        if (ImGui::Begin("Atomic Neighborhood Browser", &s_windowVisible)) {
            ImGui::Text("Total Atomic Clouds: %zu", totalAtomic);
            ImGui::Separator();

            ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg |
                                    ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable;

            if (ImGui::BeginTable("AtomicBrowserTable", 4, flags, ImVec2(0, 350))) {
                ImGui::TableSetupScrollFreeze(0, 1);
                ImGui::TableSetupColumn("Idx", ImGuiTableColumnFlags_WidthFixed, 55.0f);
                ImGui::TableSetupColumn("Cloud Name", ImGuiTableColumnFlags_WidthStretch, 200.0f);
                ImGui::TableSetupColumn("Atoms", ImGuiTableColumnFlags_WidthFixed, 50.0f);
                ImGui::TableSetupColumn("Actions", ImGuiTableColumnFlags_WidthFixed, 100.0f);
                ImGui::TableHeadersRow();

                ImGuiListClipper clipper;
                clipper.Begin(static_cast<int>(totalAtomic));
                while (clipper.Step()) {
                    for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
                        const size_t i = static_cast<size_t>(row);
                        ImGui::TableNextRow();

                        ImGui::TableSetColumnIndex(0);
                        ImGui::Text("%zu", i);

                        ImGui::TableSetColumnIndex(1);
                        ImGui::TextUnformatted(atomOps.getAtomicName(i).c_str());

                        ImGui::TableSetColumnIndex(2);
                       ImGui::Text("%zu", atomOps.getAtomicAtomCount(i));

                        ImGui::TableSetColumnIndex(3);
                        ImGui::PushID(row + 500000);
                        if (ImGui::Button("View")) {
                            atomOps.visualizeAtomicIndex(i);
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Hide")) {
                            atomOps.unviewAtomicIndex(i);
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