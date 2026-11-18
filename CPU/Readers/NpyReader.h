#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <cnpy.h>
#include "CPU/MolecularDatasetStore.h"

class NpyReader
{
public:
    static std::string atomicNumberToSymbol(int atomicNum)
    {
        switch (atomicNum)
        {
        case 1:  return "H";
        case 5:  return "B";
        case 6:  return "C";
        case 7:  return "N";
        case 8:  return "O";
        case 9:  return "F";
        case 15: return "P";
        case 16: return "S";
        case 17: return "Cl";
        case 35: return "Br";
        case 53: return "I";
        default: return "C";
        }
    }

    static bool loadDataset(MolecularDatasetStore &store,
                            const std::string &xyzInputPath,
                            const std::string &labelsInputPath,
                            int conformationIndex = 0)
    {
        namespace fs = std::filesystem;
        if (!fs::exists(xyzInputPath) || !fs::exists(labelsInputPath))
        {
            std::cout << "[NpyReader] Error: XYZ or Labels path does not exist.\n";
            return false;
        }

        std::vector<std::pair<fs::path, fs::path>> filePairs;

        if (fs::is_regular_file(xyzInputPath) && fs::is_regular_file(labelsInputPath))
        {
            filePairs.push_back({xyzInputPath, labelsInputPath});
        }
        else if (fs::is_directory(xyzInputPath) && fs::is_directory(labelsInputPath))
        {
            for (const auto &entry : fs::directory_iterator(xyzInputPath))
            {
                if (!entry.is_regular_file()) continue;
                auto path = entry.path();
                if (path.extension() != ".npy") continue;

                std::string stem = path.stem().string();
                fs::path labelPath = fs::path(labelsInputPath) / (stem + ".npy");
                if (fs::exists(labelPath))
                {
                    filePairs.push_back({path, labelPath});
                }
            }
        }
        else
        {
            std::cout << "[NpyReader] Error: Mismatched inputs. Both must be directories or both must be regular files.\n";
            return false;
        }

        size_t successCount = 0;
        for (const auto &[xyzPath, labelPath] : filePairs)
        {
            std::string stem = xyzPath.stem().string();
            try
            {
                // 1. Load coordinates: expected shape [Conformations, Atoms, 3]
                cnpy::NpyArray xyzArr = cnpy::npy_load(xyzPath.string());
                if (xyzArr.shape.size() != 3 || xyzArr.shape[2] != 3)
                {
                    std::cout << "[NpyReader] Warning: File " << xyzPath.filename() << " has unexpected shape (expected 3D: [C, N, 3])\n";
                    continue;
                }

                size_t numConfs = xyzArr.shape[0];
                size_t numAtoms = xyzArr.shape[1];

                // 2. Load labels
                cnpy::NpyArray labelArr = cnpy::npy_load(labelPath.string());

                // 3. Determine conformation range
                size_t startConf = 0;
                size_t endConf = numConfs;
                if (conformationIndex >= 0)
                {
                    if (static_cast<size_t>(conformationIndex) >= numConfs)
                    {
                        std::cout << "[NpyReader] Warning: Conformation index " << conformationIndex 
                                  << " out of range for " << stem << " (max " << numConfs - 1 << ")\n";
                        continue;
                    }
                    startConf = conformationIndex;
                    endConf = conformationIndex + 1;
                }

                fs::path cacheDir = fs::temp_directory_path() / "scientific_db_xyz_cache";
                fs::create_directories(cacheDir);

                for (size_t c = startConf; c < endConf; ++c)
                {
                    std::string tempXyzFilename = stem + "_conf" + std::to_string(c) + ".xyz";
                    fs::path tempXyzPath = cacheDir / tempXyzFilename;

                    std::ofstream out(tempXyzPath);
                    if (!out.is_open()) continue;

                    out << numAtoms << "\n";
                    out << "GEOM " << stem << " Conformation " << c << "\n";

                    for (size_t i = 0; i < numAtoms; ++i)
                    {
                        int atomicNum = 6;
                        size_t labelIdx = (labelArr.shape.size() == 2) ? (c * numAtoms + i) : i;

                        // Decode labels robustly
                        if (labelArr.word_size == 4)
                        {
                            float fVal = labelArr.data<float>()[labelIdx];
                            if (fVal >= 1.0f && fVal <= 118.0f)
                                atomicNum = static_cast<int>(fVal);
                            else
                                atomicNum = labelArr.data<int>()[labelIdx];
                        }
                        else if (labelArr.word_size == 8)
                        {
                            double dVal = labelArr.data<double>()[labelIdx];
                            if (dVal >= 1.0 && dVal <= 118.0)
                                atomicNum = static_cast<int>(dVal);
                            else
                                atomicNum = static_cast<int>(labelArr.data<int64_t>()[labelIdx]);
                        }
                        else if (labelArr.word_size == 1)
                        {
                            atomicNum = static_cast<int>(labelArr.data<char>()[labelIdx]);
                        }

                        std::string symbol = atomicNumberToSymbol(atomicNum);

                        // Decode coordinates robustly (supports both float32 and float64 arrays)
                        size_t coordIdx = (c * numAtoms + i) * 3;
                        float x = 0.0f, y = 0.0f, z = 0.0f;

                        if (xyzArr.word_size == 8)
                        {
                            const double* dPtr = xyzArr.data<double>();
                            x = static_cast<float>(dPtr[coordIdx + 0]);
                            y = static_cast<float>(dPtr[coordIdx + 1]);
                            z = static_cast<float>(dPtr[coordIdx + 2]);
                        }
                        else
                        {
                            const float* fPtr = xyzArr.data<float>();
                            x = fPtr[coordIdx + 0];
                            y = fPtr[coordIdx + 1];
                            z = fPtr[coordIdx + 2];
                        }

                        out << symbol << " " << x << " " << y << " " << z << "\n";
                    }
                    out.close();

                    if (store.addFile(tempXyzPath.string()))
                    {
                        successCount++;
                    }
                }
            }
            catch (const std::exception &e)
            {
                std::cout << "[NpyReader] Error processing " << stem << ": " << e.what() << "\n";
            }
        }

        std::cout << "[NpyReader] Successfully loaded " << successCount << " entries from " << filePairs.size() << " target file(s).\n";
        return true;
    }
};