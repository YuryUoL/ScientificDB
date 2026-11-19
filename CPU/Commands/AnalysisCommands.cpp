#include "AnalysisCommands.h"
#include "CPU/Visualization/AtomicMatchingViz.h"
#include "CPU/Duplicatechecker.h"
#include "CPU/Visualization/PolyscopeBridge.h"
#include "CPU/Visualization/Matchingvisualizer.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <omp.h>

#include <omp.h>

namespace Commands
{
    void registerAnalysisCommands(CommandDispatcher &ui, MolecularDatasetStore &xyzStore)
    {
        // ComputeDescriptors
        ui.registerCommand(
            "ComputeDescriptors", "",
            "Computes SDV, SRV, and PDD descriptors in parallel.",
            [&xyzStore](std::istringstream &)
            { xyzStore.computeDescriptors(); });
        ui.registerAlias("compute", "ComputeDescriptors");

        // FlushDescriptors
        ui.registerCommand(
            "FlushDescriptors", "",
            "Deletes and flushes all computed descriptors and FAISS indices.",
            [&xyzStore](std::istringstream &)
            { xyzStore.flushDescriptors(); });
        ui.registerAlias("flush", "FlushDescriptors");

        // knn
        ui.registerCommand(
            "knn", "<index> <k_or_pct> [metric: sdv/srv] [csv_file]",
            "Finds k neighbors using FAISS L2 search.",
            [&xyzStore](std::istringstream &iss)
            {
                size_t idx;
                std::string kStr = "5", metric = "srv", csvPath = "";
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
                    std::cout << "Usage: knn <index> <k_or_pct> [metric] [csv_file]\n";
                }
            });
        ui.registerAlias("findKNN", "knn");

        // chain
        ui.registerCommand(
            "chain", "<index> <k_or_pct> [metric: sdv/srv] [csv_file]",
            "Finds k neighbors restricted to previous survivors.",
            [&xyzStore](std::istringstream &iss)
            {
                size_t idx;
                std::string kStr = "5", metric = "sdv", csvPath = "";
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
                    std::cout << "Usage: chain <index> <k_or_pct> [metric] [csv_file]\n";
                }
            });
        ui.registerAlias("knn_chain", "chain");

        // pair_dist
        ui.registerCommand(
            "pair_dist", "<index1> <index2> [metric: srv/sdv/pdd/aligner] [key=val ...]",
            "Computes exact distance metric or full rigid alignment.",
            [&xyzStore](std::istringstream &iss)
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
                                opt.set(token.substr(0, eq), token.substr(eq + 1));
                        }
                        xyzStore.printPairAlignmentWithStats(idx1, idx2, opt);
                    }
                    else
                    {
                        auto tStart = std::chrono::high_resolution_clock::now();
                        float dist = xyzStore.computePairMetric(idx1, idx2, metric);
                        auto tEnd = std::chrono::high_resolution_clock::now();
                        double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
                        if (dist >= 0.0f)
                        {
                            std::cout << "\n=======================================================\n"
                                      << "  PAIRWISE METRIC COMPUTATION (" << metric << ")\n"
                                      << "=======================================================\n"
                                      << "  Computed Distance: " << std::fixed << std::setprecision(6) << dist << "\n"
                                      << "  Time Elapsed     : " << std::fixed << std::setprecision(3) << elapsedMs << " ms\n"
                                      << "=======================================================\n\n";
                        }
                    }
                }
                else
                {
                    std::cout << "Usage: pair_dist <index1> <index2> [metric] [key=val ...]\n";
                }
            },
            "  Parameters for aligner:\n"
            "    refine=<int>   : Kabsch refinement rounds (default: 0)\n"
            "    symbol=<0/1>   : Restrict alignment and matching to same-symbol atoms (default: 0)\n"
            "    accept=<float> : Accept/stop threshold distance (default: 0.0)\n"
            "    max=<int>      : Max candidate pairs to evaluate, 0 = all (default: 0)\n"
            "    exact=<0/1>    : 1 = find optimal minimum; 0 = return any / early-exit (default: 1)\n\n"
            "  Example:\n"
            "    pair_dist 4210 4214 aligner refine=4 symbol=1 exact=1");
        ui.registerAlias("pairDist", "pair_dist");
        ui.registerAlias("dist_pair", "pair_dist");

        // dup_check
        ui.registerCommand(
            "dup_check", "<index|all> [epsilon|csv_file] [key=val ...]",
            "Checks for molecular duplicates against the DB.",
            [&xyzStore](std::istringstream &iss)
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
                        else if (token.find(".csv") != std::string::npos)
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
                        else if (token.find(".csv") != std::string::npos)
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
                    DupRunStats st;
                    auto matches = xyzStore.runDuplicateCheck(idx, csvPath, p, st);
                    DuplicateChecker::printReport(idx, p, st, matches);
                }
            },
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
            "    parallel=<0/1> : Enable/disable multi-threaded candidate evaluation (default: 1)\n\n"
            "  Runtime Usage Examples:\n"
            "    dup_check 0 0.05 refine=4 label=1 alignf=1.5\n"
            "    dup_check all duplicates.csv 0.05 refine=4 parallel=1");
        ui.registerAlias("duplicateCheck", "dup_check");
        ui.registerAlias("dupCheck", "dup_check");

        // Register Command: knn_all <k_or_pct> <metric> <csv_file>
        ui.registerCommand(
            "knn_all", "<k_or_pct> <metric: sdv/srv> <csv_file>",
            "Computes k-nearest neighbors for ALL molecules and prints terminal tables + CSV.",
            [&xyzStore](std::istringstream &iss)
            {
                std::string kStr = "100%", metric = "srv", csvPath = "";
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

        // Register Command: dist_summary
        ui.registerCommand(
            "dist_summary", "",
            "Prints distance distribution stats, a 10-bin ASCII histogram, and outliers for the last search.",
            [&xyzStore](std::istringstream &)
            {
                xyzStore.printDistanceDistribution();
            });
        ui.registerAlias("dist_stats", "dist_summary");
        ui.registerAlias("histogram", "dist_summary");

        // Register Command: view <index>
        ui.registerCommand(
            "view", "<index>",
            "Renders a specific loaded molecule as a colored point cloud in the Polyscope viewer.",
            [&xyzStore](std::istringstream &iss)
            {
                size_t idx;
                if (iss >> idx)
                    xyzStore.visualizeIndex(idx);
                else
                    std::cout << "Usage: view <index>\n";
            });
        ui.registerAlias("visualize", "view");

        // Register Command: unview <index>
        ui.registerCommand(
            "unview", "<index>",
            "Removes a specific rendered molecule point cloud from the Polyscope viewer.",
            [&xyzStore](std::istringstream &iss)
            {
                size_t idx;
                if (iss >> idx)
                    xyzStore.unviewIndex(idx);
                else
                    std::cout << "Usage: unview <index>\n";
            });
        ui.registerAlias("hide", "unview");
        ui.registerAlias("unvisualize", "unview");

        // Register Command: gizmo on/off
        ui.registerCommand(
            "gizmo", "<on/off>",
            "Enables or disables interactive transform gizmos for the point clouds.",
            [](std::istringstream &iss)
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

        // Pair matching visualization commands
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
                [&xyzStore](size_t a, size_t b)
                { xyzStore.setMatchVisible(a, b, true); });
        pairCmd("hide_match", "Hides the matching lines.",
                [&xyzStore](size_t a, size_t b)
                { xyzStore.setMatchVisible(a, b, false); });
        pairCmd("rm_match", "Deletes the stored matching for a pair from memory.",
                [&xyzStore](size_t a, size_t b)
                { xyzStore.removeMatch(a, b); });
        pairCmd("apply_align", "Snaps cloud A onto cloud B using the stored rotation.",
                [&xyzStore](size_t a, size_t b)
                { xyzStore.applyAlignment(a, b); });

        ui.registerCommand("list_matches", "", "Lists all stored matchings.",
                           [](std::istringstream &)
                           { MatchingViz::list(); });
        ui.registerCommand("clear_matches", "", "Deletes all stored matchings.",
                           [](std::istringstream &)
                           { MatchingViz::clearAll(); });
        ui.registerCommand("show_matches", "", "Shows all stored matchings.",
                           [&xyzStore](std::istringstream &)
                           { xyzStore.setAllMatchesVisible(true); });
        ui.registerCommand("hide_matches", "", "Hides all stored matchings.",
                           [&xyzStore](std::istringstream &)
                           { xyzStore.setAllMatchesVisible(false); });

        // Cluster & Network Graph Commands
        ui.registerCommand(
            "explore_clusters", "",
            "Computes connected components from the last duplicate check and opens an interactive table window.",
            [&xyzStore](std::istringstream &)
            { xyzStore.exploreClusters(); });
        ui.registerAlias("clusters", "explore_clusters");

        ui.registerCommand("show_graph", "", "Shows the duplicate network graph in Polyscope.",
                           [&xyzStore](std::istringstream &)
                           {
                               xyzStore.setGraphVisible(true);
                               std::cout << "[Polyscope] Duplicate graph shown.\n";
                           });
        ui.registerCommand("hide_graph", "", "Hides the duplicate network graph in Polyscope.",
                           [&xyzStore](std::istringstream &)
                           {
                               xyzStore.setGraphVisible(false);
                               std::cout << "[Polyscope] Duplicate graph hidden.\n";
                           });
        ui.registerCommand("clear_graph", "", "Removes the duplicate network graph and its labels.",
                           [&xyzStore](std::istringstream &)
                           { xyzStore.clearGraph(); });
        ui.registerAlias("rm_graph", "clear_graph");
        // ComputeAtomicDescriptors
        ui.registerCommand(
            "ComputeAtomicDescriptors", "",
            "Computes centroid distance descriptors and sorted index arrays for all atomic clouds.",
            [&xyzStore](std::istringstream &)
            { xyzStore.computeAtomicDescriptors(); });
        ui.registerAlias("compute_atoms", "ComputeAtomicDescriptors");

        // atomic_dist <index1> <index2> [metric: srv/aligner] [N=val] [key=val ...]
        ui.registerCommand(
            "atomic_dist", "<index1> <index2> [metric: srv/aligner] [N=val] [key=val ...]",
            "Computes Linf distance or rigid subset alignment between two atomic clouds.",
            [&xyzStore](std::istringstream &iss)
            {
                size_t idx1, idx2;
                if (iss >> idx1 >> idx2)
                {
                    std::string metric = "srv";
                    size_t N = 0;
                    AlignmentOptions opt;
                    bool metricSet = false;

                    std::string token;
                    while (iss >> token)
                    {
                        size_t eq = token.find('=');
                        if (eq != std::string::npos)
                        {
                            std::string key = token.substr(0, eq);
                            std::string val = token.substr(eq + 1);
                            if (key == "N" || key == "n")
                            {
                                try
                                {
                                    N = std::stoul(val);
                                }
                                catch (...)
                                {
                                }
                            }
                            else
                            {
                                opt.set(key, val);
                            }
                        }
                        else if (!metricSet)
                        {
                            metric = token;
                            metricSet = true;
                        }
                    }

                    if (metric == "aligner")
                    {
                        xyzStore.computeAtomicAlignerDistance(idx1, idx2, N, opt);
                    }
                    else
                    {
                        auto tStart = std::chrono::high_resolution_clock::now();
                        float dist = xyzStore.computeAtomicPairDistance(idx1, idx2, metric, N);
                        auto tEnd = std::chrono::high_resolution_clock::now();
                        double elapsedMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
                        if (dist >= 0.0f)
                        {
                            std::cout << "\n=======================================================\n"
                                      << "  ATOMIC PAIRWISE L_INF DISTANCE (" << metric << ", N=" << (N == 0 ? "ALL" : std::to_string(N)) << ")\n"
                                      << "=======================================================\n"
                                      << "  Computed Distance: " << std::fixed << std::setprecision(6) << dist << "\n"
                                      << "  Time Elapsed     : " << std::fixed << std::setprecision(3) << elapsedMs << " ms\n"
                                      << "=======================================================\n\n";
                        }
                    }
                }
                else
                {
                    std::cout << "Usage: atomic_dist <index1> <index2> [metric: srv/aligner] [N=val] [key=val ...]\n";
                }
            },
            "  Parameters for aligner:\n"
            "    N=<int>        : Number of closest non-center neighbors from A (0 = all; clamped to |B| if larger)\n"
            "    refine=<int>   : Kabsch refinement rounds (default: 0)\n"
            "    symbol=<0/1>   : Restrict alignment and matching to same-symbol atoms (default: 0)\n"
            "    accept=<float> : Accept/stop threshold distance (default: 0.0)\n"
            "    max=<int>      : Max candidate pairs to evaluate, 0 = all (default: 0)\n"
            "    exact=<0/1>    : 1 = find optimal minimum; 0 = return any / early-exit (default: 1)\n\n"
            "  Example:\n"
            "    atomic_dist 1 6 aligner N=4 refine=4 symbol=1 exact=1");

        ui.registerAlias("atomicDist", "atomic_dist");

        ui.registerCommand("apply_atomic_align", "<idxA> <idxB>", "Rotates atomic cloud A onto B using the computed alignment.",
                           [&xyzStore](std::istringstream &iss)
                           {
                               size_t a, b;
                               if (iss >> a >> b)
                                   AtomicMatchingViz::applyAlignment(a, b);
                               else
                                   std::cout << "Usage: apply_atomic_align <idxA> <idxB>\n";
                           });

        ui.registerCommand("show_atomic_match", "<idxA> <idxB>", "Shows subset matching lines between two atomic clouds.",
                           [&xyzStore](std::istringstream &iss)
                           {
                               size_t a, b;
                               if (iss >> a >> b)
                                   AtomicMatchingViz::setVisible(a, b, true);
                               else
                                   std::cout << "Usage: show_atomic_match <idxA> <idxB>\n";
                           });
        ui.registerCommand("hide_atomic_match", "<idxA> <idxB>", "Hides subset matching lines.",
                           [&xyzStore](std::istringstream &iss)
                           {
                               size_t a, b;
                               if (iss >> a >> b)
                                   AtomicMatchingViz::setVisible(a, b, false);
                               else
                                   std::cout << "Usage: hide_atomic_match <idxA> <idxB>\n";
                           });

        ui.registerCommand(
        "atomic_classify", "[all] [epsilon|csv_file] [N=val] [key=val ...]",
        "Runs batch classification across all atomic neighborhood clouds.",
        [&xyzStore](std::istringstream &iss)
        {
            std::string csvPath = "atomic_matches.csv";
            AtomicClassifyParams p;
            std::string token;

            while (iss >> token)
            {
                if (token == "all")
                    continue; // optional syntactic sugar matching dup_check

                size_t eq = token.find('=');
                if (eq != std::string::npos)
                {
                    std::string key = token.substr(0, eq);
                    std::string val = token.substr(eq + 1);

                    // Allow explicit csv=... or out=... syntax
                    if (key == "csv" || key == "out" || key == "file")
                    {
                        csvPath = val;
                    }
                    else
                    {
                        p.set(key, val);
                    }
                }
                else if (token.find(".csv") != std::string::npos)
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

            // -------------------------------------------------------------
            // PRE-LAUNCH PARAMETER VERIFICATION BANNER
            // -------------------------------------------------------------
            std::cout << "\n=======================================================\n"
                      << "  LAUNCHING ATOMIC BATCH CLASSIFICATION\n"
                      << "=======================================================\n"
                      << "  CSV Export Path     : " << (csvPath.empty() ? "(none / disabled)" : csvPath) << "\n"
                      << "  Epsilon (epsilon)   : " << std::fixed << std::setprecision(4) << p.epsilon << " A\n"
                      << "  Neighbors (N)       : " << (p.N == 0 ? "ALL neighbors" : std::to_string(p.N)) << "\n"
                      << "  Consider Same Type  : " << (p.considerSame ? "true (all pairs)" : "false (different center types only)") << "\n"
                      << "  Optimal Aligner     : " << (p.optimal ? "true (exact bottleneck)" : "false (fast early-exit <= eps)") << "\n"
                      << "  Species Constraint  : " << (p.requireSameNeighborSpecies ? "true (elements must match)" : "false") << "\n"
                      << "  Kabsch Refinement   : " << p.refineIterations << " rounds\n"
                      << "  SRV Factor (srvf)   : " << std::fixed << std::setprecision(2) << p.srvFactor << "x (limit: " << (p.srvFactor * p.epsilon) << " A)\n"
                      << "  Align Factor (alignf: " << std::fixed << std::setprecision(2) << p.alignFactor << "x (limit: " << (p.alignFactor * p.epsilon) << " A)\n"
                      << "  Slack Tol (tol)     : " << p.relTol << "\n"
                      << "  Max Matches Stored  : " << (p.maxMatches == 0 ? "ALL" : std::to_string(p.maxMatches)) << "\n"
                      << "  Execution Mode      : " << (p.parallel ? "Parallel (OpenMP)" : "Single-threaded")
                      << " (threads: " << (p.numThreads > 0 ? std::to_string(p.numThreads) : "auto/max") << ")\n"
                      << "=======================================================\n\n" << std::flush;

            AtomicClassifyStats st;
            auto matches = xyzStore.runBatchAtomicClassify(csvPath, p, st);
            AtomicClassifier::printBatchReport(p, st, matches, csvPath);
        },
        "  Parameters & Runtime Options:\n"
        "    epsilon        : Max bottleneck distance threshold in Angstroms (default: 0.10)\n"
        "    N=<int>        : Number of nearest non-center neighbor atoms to compare (default: 4)\n"
        "    same=<0/1>     : 1 = compare all pairs; 0 = only compare different center types (default: 0)\n"
        "    optimal=<0/1>  : 1 = exact minimum bottleneck; 0 = fast early-exit aligner (default: 0)\n"
        "    species=<0/1>  : Require matched neighbor atoms to have identical element symbols (default: 0)\n"
        "    refine=<int>   : Kabsch alignment refinement rounds (default: 0)\n"
        "    srvf=<float>   : Stage 1 SRV L_inf radius multiplier factor (default: 1.0)\n"
        "    alignf=<float> : Stage 2 Aligner bottleneck multiplier factor (default: 1.0)\n"
        "    tol=<float>    : Floating-point noise slack tolerance (default: 1e-4)\n"
        "    max=<int>      : Keep only top K best matches, 0 = all (default: 0)\n"
        "    threads=<int>  : OpenMP thread count override, 0 = max available (default: 0)\n"
        "    parallel=<0/1> : Enable/disable multi-threaded candidate evaluation (default: 1)\n\n"
        "  Runtime Usage Examples:\n"
        "    atomic_classify 0.10 N=4\n"
        "    atomic_classify 0.08 my_custom_output.csv N=6 same=1 species=1 refine=2\n"
        "    atomic_classify all 0.05 csv=results.csv optimal=1 max=50 parallel=1");

        ui.registerAlias("atomicClassify", "atomic_classify");
        ui.registerAlias("classify_atoms", "atomic_classify");
        ui.registerAlias("classifyAtoms", "atomic_classify");
    }
} // namespace Commands