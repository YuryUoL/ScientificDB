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

#if USE_MYSQL
#include "Database/DatabaseConnection.h"
#include "Database/MoleculeRepository.h"
// Global or session database handle for the CLI client
static DatabaseConnection g_dbConn;
#endif

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

    MatchingViz::install();

    MolecularDatasetStore xyzStore;
    CommandDispatcher ui;

    // Register the ImGui window callback here
    PolyscopeBridge::s_extraUserCallback = [&]()
    {
        std::vector<std::string> filenames(xyzStore.size());
        std::vector<size_t> atomCounts(xyzStore.size());
        MolecularGraphViz::drawImGuiWindow(filenames, atomCounts);
    };

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

    // Register Command: loadAndCompute <file_or_folder>
    ui.registerCommand(
        "loadAndCompute", "<path_to_file_or_folder>",
        "Loads single .xyz file or folder and automatically computes all descriptors.",
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
                std::cout << "Usage: loadAndCompute <file_or_folder_path>\n";
                return;
            }
            namespace fs = std::filesystem;
            bool loaded = false;
            if (fs::is_directory(path))
            {
                xyzStore.addFolder(path);
                loaded = true;
            }
            else if (fs::is_regular_file(path))
            {
                if (xyzStore.addFile(path))
                {
                    std::cout << "[XyzStore] Successfully loaded 1 file into host RAM.\n";
                    loaded = true;
                }
            }
            else
            {
                std::cout << "[XyzStore] Error: Path does not exist or is invalid: " << path << "\n";
            }

            if (loaded)
            {
                std::cout << "[XyzStore] Automatically computing descriptors...\n";
                xyzStore.computeDescriptors();
            }
        });
    ui.registerAlias("load_and_compute", "loadAndCompute");
    ui.registerAlias("loadCompute", "loadAndCompute");

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

    ui.registerCommand(
        "pair_dist", "<index1> <index2> [metric: srv/sdv/pdd/aligner] [key=val ...]",
        "Computes exact distance metric or full rigid alignment with bottleneck distance and point matching.\n"
        "  Parameters for aligner:\n"
        "    refine=<int>   : Kabsch refinement rounds (default: 0)\n"
        "    symbol=<0/1>   : Restrict alignment and matching to same-symbol atoms (default: 0)\n"
        "    accept=<float> : Accept/stop threshold distance (default: 0.0)\n"
        "    max=<int>      : Max candidate pairs to evaluate, 0 = all (default: 0)\n"
        "    exact=<0/1>    : 1 = find optimal minimum; 0 = return any / early-exit (default: 1)\n"
        "  Example:\n"
        "    pair_dist 4210 4214 aligner refine=4 symbol=1 exact=1",
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
                    AlignmentOptions opt;
                    std::string token;
                    while (iss >> token)
                    {
                        size_t eq = token.find('=');
                        if (eq != std::string::npos)
                        {
                            opt.set(token.substr(0, eq), token.substr(eq + 1));
                        }
                    }
                    xyzStore.printPairAlignment(idx1, idx2, opt);
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
                std::cout << "Usage: pair_dist <index1> <index2> [metric: srv/sdv/pdd/aligner] [key=val ...]\n";
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

    ui.registerCommand("show_matches", "", "Shows all stored matchings.",
                       [&](std::istringstream &)
                       { xyzStore.setAllMatchesVisible(true); });
    ui.registerCommand("hide_matches", "", "Hides all stored matchings.",
                       [&](std::istringstream &)
                       { xyzStore.setAllMatchesVisible(false); });

    // ==========================================
    // 5. SYSTEM & GENERAL COMMANDS
    // ==========================================

    // Register Command: dup_check <index|all> [epsilon|csv_file] [key=val ...]
    ui.registerCommand(
        "dup_check", "<index|all> [epsilon|csv_file] [key=val ...]",
        "Checks if molecule <index> or ALL molecules are duplicates against the DB.\n"
        "  Parameters & Runtime Options:\n"
        "    epsilon        : Max per-atom displacement threshold (default: 0.05)\n"
        "    srvf=<float>   : Stage 1 SRV radius multiplier factor (default: 1.0)\n"
        "    sdvf=<float>   : Stage 2 SDV radius multiplier factor (default: 2.0)\n"
        "    pddf=<float>   : Stage 3 PDD radius multiplier factor (default: 2.0)\n"
        "    alignf=<float> : Stage 4 Aligned radius multiplier factor (default: 1.0)\n"
        "    tol=<float>    : Floating-point noise slack tolerance (default: 1e-4)\n"
        "    comp=<0/1>     : Require identical element composition multiset (default: 1)\n"
        "    label=<0/1>    : Require identical comment/label line (default: 0)\n"
        "    species=<0/1>  : Drop matches with mismatched element pairing (default: 0)\n"
        "    exact=<0/1>    : True = find true minimum; False = fast early-exit (default: 1)\n"
        "    refine=<int>   : Kabsch alignment refinement iterations (default: 0, disabled)\n"
        "    max=<int>      : Keep only top N best matches, 0 = all (default: 0)\n"
        "    threads=<int>  : OpenMP thread count override, 0 = max available (default: 0)\n"
        "    parallel=<0/1> : Enable/disable multi-threaded candidate evaluation (default: 1)\n"
        "  Runtime Usage Examples:\n"
        "    dup_check 0 0.05 refine=4 label=1 alignf=1.5\n"
        "    dup_check all duplicates.csv 0.05 refine=4 parallel=1",
        [&](std::istringstream &iss)
        {
            std::string arg1;
            if (!(iss >> arg1))
            {
                std::cout << "Usage: dup_check <index|all> [epsilon|csv_file] [key=val ...]\n";
                return;
            }

            if (arg1 == "all")
            {
                std::string csvPath = "duplicates.csv";
                DupCheckParams p;

                std::string token;
                while (iss >> token)
                {
                    size_t eq = token.find('=');
                    if (eq != std::string::npos)
                    {
                        p.set(token.substr(0, eq), token.substr(eq + 1));
                    }
                    else
                    {
                        if (token.find('/') != std::string::npos ||
                            token.find('\\') != std::string::npos ||
                            token.find(".csv") != std::string::npos)
                        {
                            csvPath = token;
                        }
                        else
                        {
                            try
                            {
                                p.epsilon = std::stof(token);
                            }
                            catch (...)
                            {
                            }
                        }
                    }
                }

                std::cout << "\n[DupCheck] Running batch duplicate check with parameters:\n"
                          << "  epsilon    : " << p.epsilon << "\n"
                          << "  exact      : " << (p.exactDistance ? "1 (optimal)" : "0 (early-exit)") << "\n"
                          << "  alignf     : " << p.alignFactor << "\n"
                          << "  srvf       : " << p.srvFactor << "\n"
                          << "  sdvf       : " << p.sdvFactor << "\n"
                          << "  pddf       : " << p.pddFactor << "\n"
                          << "  refine     : " << p.refineIterations << "\n"
                          << "  threads    : " << (p.numThreads > 0 ? p.numThreads : omp_get_max_threads()) << "\n"
                          << "  csvOutput  : " << csvPath << "\n\n"
                          << std::flush;

                DupRunStats st;
                auto matches = xyzStore.runBatchDuplicateCheck(csvPath, p, st);
                DuplicateChecker::printBatchReport(p, st, matches, csvPath);
            }
            else
            {
                size_t idx = std::stoul(arg1);
                std::string csvPath = "";
                DupCheckParams p;

                std::string token;
                while (iss >> token)
                {
                    size_t eq = token.find('=');
                    if (eq != std::string::npos)
                    {
                        p.set(token.substr(0, eq), token.substr(eq + 1));
                    }
                    else
                    {
                        if (token.find('/') != std::string::npos ||
                            token.find('\\') != std::string::npos ||
                            token.find(".csv") != std::string::npos)
                        {
                            csvPath = token;
                        }
                        else
                        {
                            try
                            {
                                p.epsilon = std::stof(token);
                            }
                            catch (...)
                            {
                            }
                        }
                    }
                }

                DupRunStats st;
                auto matches = xyzStore.runDuplicateCheck(idx, csvPath, p, st);
                DuplicateChecker::printReport(idx, p, st, matches);
            }
        });
    ui.registerAlias("duplicateCheck", "dup_check");
    ui.registerAlias("dupCheck", "dup_check");

    ui.registerCommand(
        "explore_clusters", "",
        "Computes connected components from the last duplicate check and opens an interactive table window.",
        [&](std::istringstream &)
        {
            xyzStore.exploreClusters();
        });
    ui.registerAlias("clusters", "explore_clusters");

    // Register Command: show_graph / hide_graph
    ui.registerCommand(
        "show_graph", "",
        "Shows the duplicate network graph in Polyscope.",
        [&](std::istringstream &)
        {
            xyzStore.setGraphVisible(true);
            std::cout << "[Polyscope] Duplicate graph shown.\n";
        });

    ui.registerCommand(
        "hide_graph", "",
        "Hides the duplicate network graph in Polyscope.",
        [&](std::istringstream &)
        {
            xyzStore.setGraphVisible(false);
            std::cout << "[Polyscope] Duplicate graph hidden.\n";
        });

    // Register Command: clear_graph
    ui.registerCommand(
        "clear_graph", "",
        "Removes the duplicate network graph and its labels from Polyscope.",
        [&](std::istringstream &)
        {
            xyzStore.clearGraph();
        });
    ui.registerAlias("rm_graph", "clear_graph");

#if USE_MYSQL
    // Register Command: db_connect <host> <user> <pass> <db>
    ui.registerCommand(
        "db_connect", "<host> <user> <pass> <db>",
        "Connects the client to a remote or local MariaDB server.",
        [&](std::istringstream &iss)
        {
            std::string host, user, pass, db;
            if (iss >> host >> user >> pass >> db)
            {
                if (g_dbConn.connect(host, user, pass, db))
                {
                    MoleculeRepository repo(g_dbConn);
                    repo.initializeSchema();
                    std::cout << "[Database] Connected and schema verified.\n";
                }
            }
            else
            {
                std::cout << "Usage: db_connect <host> <user> <pass> <db>\n";
            }
        });

    // Register Command: db_push
    ui.registerCommand(
        "db_push", "",
        "Uploads all currently loaded local RAM molecules into MariaDB blobs.",
        [&](std::istringstream &)
        {
            if (!g_dbConn.isConnected())
            {
                std::cout << "[Database] Error: Not connected. Run 'db_connect' first.\n";
                return;
            }
            // Implementation detail: iterate through xyzStore entries and push to repo
            std::cout << "[Database] Pushing local dataset to server...\n";
            // MoleculeRepository repo(g_dbConn);
            // ... loop and repo.saveMolecule(...)
        });

    // Register Command: db_pull
    ui.registerCommand(
        "db_pull", "",
        "Downloads molecule blobs from MariaDB into local RAM store for analysis and rendering.",
        [&](std::istringstream &)
        {
            if (!g_dbConn.isConnected())
            {
                std::cout << "[Database] Error: Not connected. Run 'db_connect' first.\n";
                return;
            }
            MoleculeRepository repo(g_dbConn);
            size_t pulled = repo.loadAllIntoMemory(xyzStore);
            std::cout << "[Database] Pulled " << pulled << " molecules into local RAM.\n";
        });
#endif

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
        bool autoCompute = (argc >= 3 && (std::string(argv[2]) == "--compute" || std::string(argv[2]) == "-c"));

        namespace fs = std::filesystem;
        bool loaded = false;
        if (fs::is_directory(initialPath))
        {
            xyzStore.addFolder(initialPath);
            loaded = true;
        }
        else if (fs::is_regular_file(initialPath))
        {
            if (xyzStore.addFile(initialPath))
            {
                std::cout << "[XyzStore] Successfully loaded 1 file into host RAM.\n";
                loaded = true;
            }
        }

        if (loaded && autoCompute)
        {
            std::cout << "[XyzStore] Automatically computing descriptors on startup...\n";
            xyzStore.computeDescriptors();
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