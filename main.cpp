#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <filesystem>

// GNU Readline headers for terminal arrow key & command history support
#include <readline/readline.h>
#include <readline/history.h>

#include "CPU/CommandDispatcher.h"
#include "CPU/MolecularDatasetStore.h"
#include "CPU/PolyscopeBridge.h"

#include "CPU/Matchingvisualizer.h"

// Thread-safe command queue & synchronization primitives
static std::mutex g_cmdMutex;
static std::condition_variable g_cmdCV;
static std::queue<std::string> g_cmdQueue;
static std::atomic<bool> g_running{true};
static bool g_readyForInput{true};

/**
 * @brief Asynchronous CLI reader thread maintaining terminal line-editing & command history via GNU Readline.
 */
void inputThreadWorker()
{
    char *rawInput = nullptr;
    while (g_running)
    {
        {
            std::unique_lock<std::mutex> lock(g_cmdMutex);
            g_cmdCV.wait(lock, []
                         { return g_readyForInput || !g_running; });
            if (!g_running)
                break;
        }

        rawInput = readline("> ");
        if (!rawInput)
        { // Handles Ctrl+D (EOF)
            g_running = false;
            break;
        }

        std::string line(rawInput);
        if (!line.empty())
        {
            add_history(rawInput); // Save command into up/down arrow key history
        }
        free(rawInput);

        {
            std::lock_guard<std::mutex> lock(g_cmdMutex);
            g_cmdQueue.push(line);
            g_readyForInput = false;
        }

        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;
        if (cmd == "quit" || cmd == "exit")
        {
            g_running = false;
            break;
        }
    }
    g_running = false;
}

int main(int argc, char *argv[])
{
    std::cout << "Initializing Polyscope GUI...\n";
    PolyscopeBridge::init();

    MolecularDatasetStore xyzStore;
    CommandDispatcher ui;

    // ==========================================
    // 1. DATASET MANAGEMENT & INSPECTION
    // ==========================================

    // Register Command: loadXyz <file_or_folder>
    ui.registerCommand(
        "loadXyz", "<path_to_file_or_folder>",
        "Loads single .xyz file or a whole folder of .xyz files into host RAM.",
        [&](std::istringstream &iss)
        {
            std::string path;
            iss >> std::ws;
            if (iss.peek() == '\'' || iss.peek() == '"')
            {
                char quote = iss.get();
                std::getline(iss, path, quote);
            }
            else
            {
                iss >> path;
            }

            if (path.empty())
            {
                std::cout << "Usage: loadXyz <file_or_folder_path>\n";
                return;
            }
            namespace fs = std::filesystem;
            if (fs::is_directory(path))
            {
                xyzStore.addFolder(path);
            }
            else if (fs::is_regular_file(path))
            {
                if (xyzStore.addFile(path))
                {
                    std::cout << "[XyzStore] Successfully loaded 1 file into host RAM.\n";
                }
            }
            else
            {
                std::cout << "[XyzStore] Error: Path does not exist or is invalid: " << path << "\n";
            }
        });

    // Register Command: ShowAll [limit]
    ui.registerCommand(
        "ShowAll", "[limit]",
        "Lists all loaded .xyz molecules with general atom count info.",
        [&](std::istringstream &iss)
        {
            size_t limit = 25;
            iss >> limit;
            xyzStore.showAll(limit);
        });
    ui.registerAlias("showall", "ShowAll");
    ui.registerAlias("listXyz", "ShowAll");

    // Register Command: check <index>
    ui.registerCommand(
        "check", "<index>",
        "Inspects a specific molecule index, printing atom types and coordinates.",
        [&](std::istringstream &iss)
        {
            size_t idx;
            if (iss >> idx)
            {
                xyzStore.inspect(idx);
            }
            else
            {
                std::cout << "Usage: check <index>\n";
            }
        });
    ui.registerAlias("inspectXyz", "check");

    // Register Command: removeXyz <index>
    ui.registerCommand(
        "removeXyz", "<index>",
        "Removes a molecule by its list index.",
        [&](std::istringstream &iss)
        {
            size_t idx;
            if (iss >> idx)
            {
                xyzStore.removeByIndex(idx);
            }
            else
            {
                std::cout << "Usage: removeXyz <index>\n";
            }
        });
    ui.registerAlias("rmXyz", "removeXyz");

    // Register Command: clearXyz
    ui.registerCommand(
        "clearXyz", "",
        "Clears all loaded molecular datasets.",
        [&](std::istringstream &)
        {
            xyzStore.clearAll();
        });

    // ==========================================
    // 2. DESCRIPTOR COMPUTATION & MANAGEMENT
    // ==========================================

    // Register Command: ComputeDescriptors
    ui.registerCommand(
        "ComputeDescriptors", "",
        "Computes SDV, SRV, and PDD descriptors in parallel for all loaded molecules using OpenMP.",
        [&](std::istringstream &)
        {
            xyzStore.computeDescriptors();
        });
    ui.registerAlias("compute", "ComputeDescriptors");

    // Register Command: FlushDescriptors
    ui.registerCommand(
        "FlushDescriptors", "",
        "Deletes and flushes all computed SDV, SRV, and PDD descriptors along with FAISS indices.",
        [&](std::istringstream &)
        {
            xyzStore.flushDescriptors();
        });
    ui.registerAlias("flush", "FlushDescriptors");

    // ==========================================
    // 3. VISUALIZATION & UI
    // ==========================================

    // Register Command: view <index>
    ui.registerCommand(
        "view", "<index>",
        "Renders a specific loaded molecule as a colored point cloud in the Polyscope viewer.",
        [&](std::istringstream &iss)
        {
            size_t idx;
            if (iss >> idx)
            {
                xyzStore.visualizeIndex(idx);
            }
            else
            {
                std::cout << "Usage: view <index>\n";
            }
        });
    ui.registerAlias("visualize", "view");

    // Register Command: unview <index>
    ui.registerCommand(
        "unview", "<index>",
        "Removes a specific rendered molecule point cloud from the Polyscope viewer.",
        [&](std::istringstream &iss)
        {
            size_t idx;
            if (iss >> idx)
            {
                xyzStore.unviewIndex(idx);
            }
            else
            {
                std::cout << "Usage: unview <index>\n";
            }
        });
    ui.registerAlias("hide", "unview");
    ui.registerAlias("unvisualize", "unview");

    // Register Command: gizmo on/off
    ui.registerCommand(
        "gizmo", "<on/off>",
        "Enables or disables interactive transform gizmos for the point clouds.",
        [&](std::istringstream &iss)
        {
            std::string state;
            iss >> state;
            if (state == "on" || state == "1")
            {
                PolyscopeBridge::setGizmoEnabled(true);
                std::cout << "[Polyscope] Interactive gizmos enabled.\n";
            }
            else if (state == "off" || state == "0")
            {
                PolyscopeBridge::setGizmoEnabled(false);
                std::cout << "[Polyscope] Interactive gizmos disabled.\n";
            }
            else
            {
                std::cout << "Usage: gizmo on  OR  gizmo off\n";
            }
        });

    // ==========================================
    // 4. SEARCH, METRICS & ANALYSIS
    // ==========================================

    // Register Command: knn <index> <k_or_pct> [metric] [csv_file]
    ui.registerCommand(
        "knn", "<index> <k_or_pct> [metric: sdv/srv] [csv_file]",
        "Finds k neighbors (e.g. 10 or 10%) using FAISS L2 search and caches results.",
        [&](std::istringstream &iss)
        {
            size_t idx;
            std::string kStr = "5";
            std::string metric = "srv";
            std::string csvPath = "";

            if (iss >> idx)
            {
                if (iss >> kStr)
                {
                    iss >> metric;
                    iss >> csvPath;
                }
                xyzStore.searchKNN(idx, kStr, metric, csvPath, false);
            }
            else
            {
                std::cout << "Usage: knn <index> <k_or_pct> [metric: sdv/srv] [csv_file]\n";
            }
        });
    ui.registerAlias("findKNN", "knn");

    // Register Command: chain <index> <k_or_pct> [metric] [csv_file]
    ui.registerCommand(
        "chain", "<index> <k_or_pct> [metric: sdv/srv] [csv_file]",
        "Finds k neighbors restricted to the survivors of the previous search operation.",
        [&](std::istringstream &iss)
        {
            size_t idx;
            std::string kStr = "5";
            std::string metric = "sdv";
            std::string csvPath = "";

            if (iss >> idx)
            {
                if (iss >> kStr)
                {
                    iss >> metric;
                    iss >> csvPath;
                }
                xyzStore.searchKNN(idx, kStr, metric, csvPath, true);
            }
            else
            {
                std::cout << "Usage: chain <index> <k_or_pct> [metric: sdv/srv] [csv_file]\n";
            }
        });
    ui.registerAlias("knn_chain", "chain");

    // Register Command: knn_all <k_or_pct> <metric> <csv_file>
    ui.registerCommand(
        "knn_all", "<k_or_pct> <metric: sdv/srv> <csv_file>",
        "Computes k-nearest neighbors (e.g. 10 or 100%) for ALL molecules and prints terminal tables + CSV.",
        [&](std::istringstream &iss)
        {
            std::string kStr = "100%";
            std::string metric = "srv";
            std::string csvPath = "";
            if (iss >> kStr >> metric >> csvPath)
            {
                xyzStore.searchKNNForAll(kStr, metric, csvPath);
            }
            else
            {
                std::cout << "Usage: knn_all <k_or_pct> <sdv/srv> <csv_file_path>\n";
            }
        });
    ui.registerAlias("knnAll", "knn_all");

    // Register Command: pair_dist <idx1> <idx2> [metric]
    ui.registerCommand(
        "pair_dist", "<index1> <index2> [metric: srv/sdv/pdd/aligner]",
        "Computes exact distance metric or full rigid alignment with bottleneck distance and point matching.",
        [&](std::istringstream &iss)
        {
            size_t idx1, idx2;
            std::string metric = "pdd";
            if (iss >> idx1 >> idx2)
            {
                if (!(iss >> metric))
                    metric = "pdd";

                if (metric == "aligner")
                {
                    xyzStore.printPairAlignment(idx1, idx2);
                }
                else
                {
                    auto tStart = std::chrono::high_resolution_clock::now();
                    float dist = xyzStore.computePairMetric(idx1, idx2, metric);
                    auto tEnd = std::chrono::high_resolution_clock::now();
                    double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

                    if (dist >= 0.0f)
                    {
                        std::cout << "\n=======================================================\n";
                        std::cout << "  PAIRWISE METRIC COMPUTATION (" << metric << ")\n";
                        std::cout << "=======================================================\n";
                        std::cout << "  Molecule A Index : " << idx1 << "\n";
                        std::cout << "  Molecule B Index : " << idx2 << "\n";
                        std::cout << "  Computed Distance: " << std::fixed << std::setprecision(6) << dist << "\n";
                        std::cout << "  Time Elapsed     : " << std::fixed << std::setprecision(3) << elapsedMs << " ms\n";
                        std::cout << "=======================================================\n\n";
                    }
                }
            }
            else
            {
                std::cout << "Usage: pair_dist <index1> <index2> [metric: srv/sdv/pdd/aligner]\n";
            }
        });
    ui.registerAlias("pairDist", "pair_dist");
    ui.registerAlias("dist_pair", "pair_dist");

    // Register Command: dist_summary
    ui.registerCommand(
        "dist_summary", "",
        "Prints distance distribution stats, a 10-bin ASCII histogram, and outliers for the last search.",
        [&](std::istringstream &)
        {
            xyzStore.printDistanceDistribution();
        });
    ui.registerAlias("dist_stats", "dist_summary");
    ui.registerAlias("histogram", "dist_summary");

    auto pairCmd = [&](const std::string &name, const std::string &desc, auto fn)
    {
        ui.registerCommand(name, "<index1> <index2>", desc,
                           [fn](std::istringstream &iss)
                           {
                               size_t a, b;
                               if (iss >> a >> b)
                                   fn(a, b);
                               else
                                   std::cout << "Usage: <command> <index1> <index2>\n";
                           });
    };

    pairCmd("show_match", "Shows the stored matching lines between two molecules.",
            [&](size_t a, size_t b)
            { xyzStore.setMatchVisible(a, b, true); });
    pairCmd("hide_match", "Hides the matching lines (the matching stays in memory).",
            [&](size_t a, size_t b)
            { xyzStore.setMatchVisible(a, b, false); });
    pairCmd("rm_match", "Deletes the stored matching for a pair from memory.",
            [&](size_t a, size_t b)
            { xyzStore.removeMatch(a, b); });
    pairCmd("apply_align", "Snaps cloud A onto cloud B using the stored rotation (use the same order as pair_dist).",
            [&](size_t a, size_t b)
            { xyzStore.applyAlignment(a, b); });

    ui.registerCommand("list_matches", "", "Lists all stored matchings.",
                       [&](std::istringstream &)
                       { MatchingViz::list(); });
    ui.registerCommand("clear_matches", "", "Deletes all stored matchings.",
                       [&](std::istringstream &)
                       { MatchingViz::clearAll(); });

    // ==========================================
    // 5. SYSTEM & GENERAL COMMANDS
    // ==========================================

    // Register Command: help
    ui.registerCommand("help", "", "Displays this command usage reference.",
                       [&](std::istringstream &)
                       {
                           ui.printHelp();
                       });

    // Register Command: quit / exit
    ui.registerCommand("quit", "", "Exits the application.",
                       [&](std::istringstream &)
                       {
                           g_running = false;
                       });
    ui.registerAlias("exit", "quit");

    if (argc >= 2)
    {
        std::string initialPath = argv[1];
        namespace fs = std::filesystem;
        if (fs::is_directory(initialPath))
        {
            xyzStore.addFolder(initialPath);
        }
        else if (fs::is_regular_file(initialPath))
        {
            xyzStore.addFile(initialPath);
        }
    }

    std::cout << "\nInteractive Molecular Store CLI Active.\n";
    std::cout << "Type 'help' for commands or 'ShowAll' to view loaded entries.\n\n";

    // Launch non-blocking CLI input thread
    std::thread inputThread(inputThreadWorker);

    // Main non-blocking render & event loop
    while (g_running)
    {
        std::string line;
        bool haveLine = false;

        {
            std::lock_guard<std::mutex> lock(g_cmdMutex);
            if (!g_cmdQueue.empty())
            {
                line = g_cmdQueue.front();
                g_cmdQueue.pop();
                haveLine = true;
            }
        }

        if (haveLine)
        {
            ui.execute(line);

            {
                std::lock_guard<std::mutex> lock(g_cmdMutex);
                g_readyForInput = true;
            }
            g_cmdCV.notify_one();
        }

        // Keep Polyscope responsive continuously
        MatchingViz::update();
        PolyscopeBridge::drawFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::cout << "Cleaning up and exiting...\n";
    if (inputThread.joinable())
    {
        inputThread.detach();
    }

    return 0;
}