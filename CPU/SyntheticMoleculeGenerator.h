#pragma once

#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <iostream>
#include "MolecularDatasetStore.h"

class SyntheticMoleculeGenerator {
public:
    /**
     * @brief Generates synthetic QM9-like molecules with configurable atom sizes, isometric copies, and noise.
     * @param store Reference to MolecularDatasetStore to populate.
     * @param numBaseMolecules Number of distinct base structures to generate.
     * @param copiesPerMolecule Number of rotated/noisy isometric variants per base.
     * @param minAtoms Minimum number of atoms per molecule.
     * @param maxAtoms Maximum number of atoms per molecule.
     * @param noiseSigma Standard deviation of Gaussian-like noise added to isometric copies.
     */
    static size_t generate(MolecularDatasetStore& store, size_t numBaseMolecules = 50, size_t copiesPerMolecule = 3, 
                           size_t minAtoms = 5, size_t maxAtoms = 25, float noiseSigma = 0.01f) 
    {
        std::cout << "[SyntheticGenerator] Generating synthetic molecular dataset (Atoms: " << minAtoms << "-" << maxAtoms << ")...\n";
        
        const std::vector<std::string> atomTypes = {"C", "N", "O", "F", "H"};
        std::random_device rd;
        std::mt19937 gen(42); 
        std::uniform_int_distribution<size_t> atomCountDist(minAtoms, maxAtoms);
        std::uniform_int_distribution<size_t> typeDist(0, atomTypes.size() - 1);
        std::uniform_real_distribution<float> coordDist(-3.0f, 3.0f);
        std::uniform_real_distribution<float> noiseDist(-noiseSigma, noiseSigma);
        std::uniform_real_distribution<float> rotDist(-3.14159f, 3.14159f);

        size_t totalGenerated = 0;

        for (size_t b = 0; b < numBaseMolecules; ++b) {
            size_t numAtoms = atomCountDist(gen);
            std::vector<AtomRecord> baseAtoms(numAtoms);
            for (size_t i = 0; i < numAtoms; ++i) {
                baseAtoms[i] = {
                    atomTypes[typeDist(gen)],
                    coordDist(gen),
                    coordDist(gen),
                    coordDist(gen),
                    0.0
                };
            }

            std::string baseFilename = "synth_mol_" + std::to_string(b) + "_base.xyz";
            std::string baseComment = "Generated QM9-like base molecule " + std::to_string(b);
            
            store.addSyntheticEntry(baseFilename, baseComment, baseAtoms);
            totalGenerated++;

            float rx = rotDist(gen), ry = rotDist(gen), rz = rotDist(gen);
            float cx = std::cos(rx), sx = std::sin(rx);
            float cy = std::cos(ry), sy = std::sin(ry);
            float cz = std::cos(rz), sz = std::sin(rz);

            for (size_t c = 0; c < copiesPerMolecule; ++c) {
                std::vector<AtomRecord> copyAtoms(numAtoms);
                for (size_t i = 0; i < numAtoms; ++i) {
                    float x = baseAtoms[i].x;
                    float y = baseAtoms[i].y;
                    float z = baseAtoms[i].z;

                    float y1 = y * cx - z * sx;
                    float z1 = y * sx + z * cx;
                    float x1 = x;

                    float x2 = x1 * cy + z1 * sy;
                    float z2 = -x1 * sy + z1 * cy;
                    float y2 = y1;

                    float xFinal = x2 * cz - y2 * sz + noiseDist(gen);
                    float yFinal = x2 * sz + y2 * cz + noiseDist(gen);
                    float zFinal = z2 + noiseDist(gen);

                    copyAtoms[i] = {
                        baseAtoms[i].symbol,
                        xFinal,
                        yFinal,
                        zFinal,
                        baseAtoms[i].charge
                    };
                }

                std::string copyFilename = "synth_mol_" + std::to_string(b) + "_copy_" + std::to_string(c) + ".xyz";
                std::string copyComment = "Isometric copy " + std::to_string(c) + " of base " + std::to_string(b);

                store.addSyntheticEntry(copyFilename, copyComment, copyAtoms);
                totalGenerated++;
            }
        }

        std::cout << "[SyntheticGenerator] Successfully generated and added " << totalGenerated << " synthetic molecules.\n";
        return totalGenerated;
    }
};