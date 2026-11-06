#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <map>
#include <memory>
#include <omp.h>
#include <faiss/IndexFlat.h>
#include "PolyscopeBridge.h"

struct MolecularDescriptors {
    size_t numAtoms = 0;
    std::vector<float> sdv; // Exact natural size: N*(N-1)/2 (Sorted pairwise distances)
    std::vector<float> srv; // Exact natural size: N (Sorted center distances)
    bool computed = false;
};

class FaissIndexManager {
public:
    FaissIndexManager() = default;
    ~FaissIndexManager() { clear(); }

    template <typename GetAtomsFunc>
    void computeDescriptorsForAll(size_t numEntries, GetAtomsFunc getAtoms) {
        if (numEntries == 0) {
            std::cout << "[FaissManager] No molecules loaded to compute descriptors.\n";
            return;
        }

        m_descriptors.assign(numEntries, MolecularDescriptors{});

        double startTime = omp_get_wtime();
        std::cout << "[FaissManager] Computing exact natural-size SDV and SRV descriptors in parallel...\n";

        #pragma omp parallel for schedule(dynamic, 16)
        for (int i = 0; i < static_cast<int>(numEntries); ++i) {
            const auto& atoms = getAtoms(i);
            MolecularDescriptors desc;
            size_t n = atoms.size();
            desc.numAtoms = n;

            if (n == 0) {
                desc.computed = true;
                m_descriptors[i] = std::move(desc);
                continue;
            }

            // 1. Compute Geometric Center (Mean position)
            float cx = 0.0f, cy = 0.0f, cz = 0.0f;
            for (const auto& a : atoms) {
                cx += a.x;
                cy += a.y;
                cz += a.z;
            }
            cx /= n;
            cy /= n;
            cz /= n;

            // 2. Compute SDV (Sorted Pairwise Distances, size N*(N-1)/2)
            size_t numPairs = (n > 1) ? (n * (n - 1)) / 2 : 0;
            std::vector<float> pairwiseDistances;
            pairwiseDistances.reserve(numPairs);

            for (size_t a = 0; a < n; ++a) {
                for (size_t b = a + 1; b < n; ++b) {
                    float dx = atoms[b].x - atoms[a].x;
                    float dy = atoms[b].y - atoms[a].y;
                    float dz = atoms[b].z - atoms[a].z;
                    pairwiseDistances.push_back(std::sqrt(dx*dx + dy*dy + dz*dz));
                }
            }
            std::sort(pairwiseDistances.begin(), pairwiseDistances.end());
            desc.sdv = std::move(pairwiseDistances);

            // 3. Compute SRV (Sorted Center Distances, size N)
            std::vector<float> centerDistances;
            centerDistances.reserve(n);

            for (const auto& a : atoms) {
                float dx = a.x - cx;
                float dy = a.y - cy;
                float dz = a.z - cz;
                centerDistances.push_back(std::sqrt(dx*dx + dy*dy + dz*dz));
            }
            std::sort(centerDistances.begin(), centerDistances.end());
            desc.srv = std::move(centerDistances);

            desc.computed = true;
            m_descriptors[i] = std::move(desc);
        }

        double elapsedTime = omp_get_wtime() - startTime;
        std::cout << "[FaissManager] Successfully computed descriptors in " 
                  << std::fixed << std::setprecision(3) << elapsedTime << " seconds.\n";

        buildIndices();
    }

    void buildIndices() {
        if (m_descriptors.empty()) return;
        m_sdvIndices.clear();
        m_srvIndices.clear();

        // Group molecules by their atom count so each group has a uniform, natural dimension
        std::map<size_t, std::vector<size_t>> atomGroups;
        for (size_t i = 0; i < m_descriptors.size(); ++i) {
            if (m_descriptors[i].computed) {
                atomGroups[m_descriptors[i].numAtoms].push_back(i);
            }
        }

        for (const auto& [nAtoms, indices] : atomGroups) {
            if (indices.empty() || nAtoms == 0) continue;

            size_t numPairs = (nAtoms > 1) ? (nAtoms * (nAtoms - 1)) / 2 : 0;
            size_t sdvDim = numPairs;
            size_t srvDim = nAtoms;

            // Build SDV index for this atom count group
            if (sdvDim > 0) {
                auto sdvIdx = std::make_unique<faiss::IndexFlatL2>(sdvDim);
                std::vector<float> sdvFlat;
                sdvFlat.reserve(indices.size() * sdvDim);
                for (size_t idx : indices) {
                    sdvFlat.insert(sdvFlat.end(), m_descriptors[idx].sdv.begin(), m_descriptors[idx].sdv.end());
                }
                sdvIdx->add(indices.size(), sdvFlat.data());
                m_sdvIndices[nAtoms] = std::move(sdvIdx);
            }

            // Build SRV index for this atom count group
            auto srvIdx = std::make_unique<faiss::IndexFlatL2>(srvDim);
            std::vector<float> srvFlat;
            srvFlat.reserve(indices.size() * srvDim);
            for (size_t idx : indices) {
                srvFlat.insert(srvFlat.end(), m_descriptors[idx].srv.begin(), m_descriptors[idx].srv.end());
            }
            srvIdx->add(indices.size(), srvFlat.data());
            m_srvIndices[nAtoms] = std::move(srvIdx);
        }

        std::cout << "[FaissManager] FAISS multi-indices built successfully across " << m_srvIndices.size() << " atom-count groups.\n";
    }

    void clear() {
        m_descriptors.clear();
        m_sdvIndices.clear();
        m_srvIndices.clear();
        std::cout << "[FaissManager] All descriptors and FAISS indices flushed/deleted.\n";
    }

    void printDescriptors(size_t index) const {
        if (index >= m_descriptors.size() || !m_descriptors[index].computed) {
            std::cout << "  FAISS Descriptors: [Not computed / Flushed. Run 'ComputeDescriptors' first]\n";
            return;
        }

        const auto& desc = m_descriptors[index];
        std::cout << "  -----------------------------------------------------\n";
        std::cout << "  FAISS Descriptors (Atoms: " << desc.numAtoms << " | SDV Dim: " << desc.sdv.size() << ", SRV Dim: " << desc.srv.size() << ")\n";
        
        std::cout << "  SDV Vector (" << desc.sdv.size() << " elements):\n    [ ";
        for (float val : desc.sdv) {
            std::cout << std::fixed << std::setprecision(3) << val << " ";
        }
        std::cout << "]\n";

        std::cout << "  SRV Vector (" << desc.srv.size() << " elements):\n    [ ";
        for (float val : desc.srv) {
            std::cout << std::fixed << std::setprecision(3) << val << " ";
        }
        std::cout << "]\n";
        std::cout << "  -----------------------------------------------------\n";
    }

private:
    std::vector<MolecularDescriptors> m_descriptors;
    std::map<size_t, std::unique_ptr<faiss::IndexFlatL2>> m_sdvIndices;
    std::map<size_t, std::unique_ptr<faiss::IndexFlatL2>> m_srvIndices;
};