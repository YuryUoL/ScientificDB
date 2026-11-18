#include "DatasetCommands.h"
#include "CPU/Readers/NpyReader.h"
#include "CPU/SyntheticMoleculeGenerator.h"
#include <filesystem>
#include <iostream>

namespace Commands {
void registerDatasetCommands(CommandDispatcher &ui, MolecularDatasetStore &xyzStore) {
    // loadXyz
    ui.registerCommand(
        "loadXyz", "<path_to_file_or_folder>",
        "Loads single .xyz file or a whole folder of .xyz files into host RAM.",
        [&xyzStore](std::istringstream &iss) {
            std::string path;
            iss >> std::ws;
            if (iss.peek() == '\'' || iss.peek() == '"') {
                char quote = iss.get();
                std::getline(iss, path, quote);
            } else {
                iss >> path;
            }

            if (path.empty()) {
                std::cout << "Usage: loadXyz <file_or_folder_path>\n";
                return;
            }
            namespace fs = std::filesystem;
            if (fs::is_directory(path)) {
                xyzStore.addFolder(path);
            } else if (fs::is_regular_file(path)) {
                if (xyzStore.addFile(path)) {
                    std::cout << "[XyzStore] Successfully loaded 1 file into host RAM.\n";
                }
            } else {
                std::cout << "[XyzStore] Error: Path does not exist or is invalid: " << path << "\n";
            }
        });

    // loadAndCompute
    ui.registerCommand(
        "loadAndCompute", "<path_to_file_or_folder>",
        "Loads single .xyz file or folder and automatically computes all descriptors.",
        [&xyzStore](std::istringstream &iss) {
            std::string path;
            iss >> std::ws;
            if (iss.peek() == '\'' || iss.peek() == '"') {
                char quote = iss.get();
                std::getline(iss, path, quote);
            } else {
                iss >> path;
            }

            if (path.empty()) {
                std::cout << "Usage: loadAndCompute <file_or_folder_path>\n";
                return;
            }
            namespace fs = std::filesystem;
            bool loaded = false;
            if (fs::is_directory(path)) {
                xyzStore.addFolder(path);
                loaded = true;
            } else if (fs::is_regular_file(path)) {
                if (xyzStore.addFile(path)) {
                    std::cout << "[XyzStore] Successfully loaded 1 file into host RAM.\n";
                    loaded = true;
                }
            } else {
                std::cout << "[XyzStore] Error: Path does not exist or is invalid: " << path << "\n";
            }

            if (loaded) {
                std::cout << "[XyzStore] Automatically computing descriptors...\n";
                xyzStore.computeDescriptors();
            }
        });
    ui.registerAlias("load_and_compute", "loadAndCompute");
    ui.registerAlias("loadCompute", "loadAndCompute");

    // loadNpy
    ui.registerCommand(
        "loadNpy", "<xyz_folder> <labels_folder> [conformation_index]",
        "Loads GEOM .npy dataset folders (XYZ coordinates & atomic labels) into host RAM.",
        [&xyzStore](std::istringstream &iss) {
            std::string xyzFolder, labelsFolder;
            int confIdx = 0;

            iss >> std::ws;
            if (iss.peek() == '\'' || iss.peek() == '"') {
                char q = iss.get();
                std::getline(iss, xyzFolder, q);
            } else {
                iss >> xyzFolder;
            }

            iss >> std::ws;
            if (iss.peek() == '\'' || iss.peek() == '"') {
                char q = iss.get();
                std::getline(iss, labelsFolder, q);
            } else {
                iss >> labelsFolder;
            }

            if (iss >> confIdx) {}

            if (xyzFolder.empty() || labelsFolder.empty()) {
                std::cout << "Usage: loadNpy <xyz_folder> <labels_folder> [conformation_index]\n";
                return;
            }

            NpyReader::loadDataset(xyzStore, xyzFolder, labelsFolder, confIdx);
        });
    ui.registerAlias("load_npy", "loadNpy");

    // ShowAll
    ui.registerCommand(
        "ShowAll", "[limit]",
        "Lists all loaded .xyz molecules with general atom count info.",
        [&xyzStore](std::istringstream &iss) {
            size_t limit = 25;
            iss >> limit;
            xyzStore.showAll(limit);
        });
    ui.registerAlias("showall", "ShowAll");
    ui.registerAlias("listXyz", "ShowAll");

    // check / inspect
    ui.registerCommand(
        "check", "<index>",
        "Inspects a specific molecule index, printing atom types and coordinates.",
        [&xyzStore](std::istringstream &iss) {
            size_t idx;
            if (iss >> idx) xyzStore.inspect(idx);
            else std::cout << "Usage: check <index>\n";
        });
    ui.registerAlias("inspectXyz", "check");

    // removeXyz
    ui.registerCommand(
        "removeXyz", "<index>",
        "Removes a molecule by its list index.",
        [&xyzStore](std::istringstream &iss) {
            size_t idx;
            if (iss >> idx) xyzStore.removeByIndex(idx);
            else std::cout << "Usage: removeXyz <index>\n";
        });
    ui.registerAlias("rmXyz", "removeXyz");

    // clearXyz
    ui.registerCommand(
        "clearXyz", "",
        "Clears all loaded molecular datasets.",
        [&xyzStore](std::istringstream &) { xyzStore.clearAll(); });

    // Generate synthetic
    ui.registerCommand(
        "generate", "[base_count] [copies_per_base] [min_atoms] [max_atoms] [noise_sigma]",
        "Generates synthetic QM9-like molecules.",
        [&xyzStore](std::istringstream &iss) {
            size_t baseMols = 50, copies = 3, minAtoms = 5, maxAtoms = 25;
            float noise = 0.01f;
            iss >> baseMols >> copies >> minAtoms >> maxAtoms >> noise;
            SyntheticMoleculeGenerator::generate(xyzStore, baseMols, copies, minAtoms, maxAtoms, noise);
        });

    // Expand all molecules into atomic neighborhood clouds
    ui.registerCommand(
        "expand_atoms", "",
        "Expands ALL loaded molecules into centered atomic neighborhood clouds.",
        [&xyzStore](std::istringstream &) {
            xyzStore.generateAllAtomicClouds();
        });

    // Expand a specific molecule index
    ui.registerCommand(
        "expand_atom_idx", "<molecule_index>",
        "Expands only a single specified molecule index into atomic neighborhoods.",
        [&xyzStore](std::istringstream &iss) {
            size_t idx;
            if (iss >> idx) {
                xyzStore.generateAtomicCloudsFor(idx);
                std::cout << "[AtomicStore] Expanded molecule index " << idx << ".\n";
            } else {
                std::cout << "Usage: expand_atom_idx <molecule_index>\n";
            }
        });

    // Reset/Clear atomic clouds
    ui.registerCommand(
        "clear_atoms", "",
        "Clears and deletes all generated atomic neighborhood clouds.",
        [&xyzStore](std::istringstream &) {
            xyzStore.clearAtomicClouds();
        });
}
} // namespace Commands