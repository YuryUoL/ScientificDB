#include "AnalysisCommands.h"
#include "CPU/MolecularOperations.h"
#include "CPU/AtomicOperations.h"
#include "CPU/Visualization/AtomicMatchingViz.h"
#include "CPU/Duplicatechecker.h"
#include "CPU/Visualization/PolyscopeBridge.h"
#include "CPU/Visualization/Matchingvisualizer.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include "CPU/Aligners/ProjectionBasedInvariantDistance.h"
#include <omp.h>

namespace Commands
{
    void registerAnalysisCommands(CommandDispatcher &ui, MolecularDatasetStore &xyzStore)
    {
        auto molOps = std::make_shared<MolecularOperations>(xyzStore);
        auto atomOps = std::make_shared<AtomicOperations>(xyzStore);

        // ComputeDescriptors
        ui.registerCommand(
            "ComputeDescriptors", "",
            "Computes SDV, SRV, and PDD descriptors in parallel.",
            [molOps](std::istringstream &)
            { molOps->computeDescriptors(); });
        ui.registerAlias("compute", "ComputeDescriptors");

        // FlushDescriptors
        ui.registerCommand(
            "FlushDescriptors", "",
            "Deletes and flushes all computed descriptors and FAISS indices.",
            [molOps](std::istringstream &)
            { molOps->flushDescriptors(); });
        ui.registerAlias("flush", "FlushDescriptors");

        // knn
        ui.registerCommand(
            "knn", "<index> <k_or_pct> [metric: sdv/srv] [csv_file]",
            "Finds k neighbors using FAISS L2 search.",
            [molOps](std::istringstream &iss)
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
                    molOps->searchKNN(idx, kStr, metric, csvPath, false);
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
            [molOps](std::istringstream &iss)
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
                    molOps->searchKNN(idx, kStr, metric, csvPath, true);
                }
                else
                {
                    std::cout << "Usage: chain <index> <k_or_pct> [metric] [csv_file]\n";
                }
            });
        ui.registerAlias("knn_chain", "chain");

        // pair_dist
           // pair_dist
    ui.registerCommand(
        "pair_dist", "<index1> <index2> [metric: srv/sdv/pdd/aligner/FullProjectionDistance] [key=val ...]",
        "Computes exact distance metric or full rigid alignment.",
        [molOps](std::istringstream &iss)
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
                    molOps->printPairAlignmentWithStats(idx1, idx2, opt);
                }
                else if (metric == "FullProjectionDistance")
                {
                    PbiOptions opt; // defaults: eps=0.01 (relative), normalize=1, reflect=0, same=0
                    auto toBool = [](const std::string &v)
                    {
                        return v == "1" || v == "true" || v == "on" || v == "yes";
                    };
                    auto parseEps = [&opt](const std::string &s)
                    {
                        try
                        {
                            opt.epsilon = std::stof(s);
                        }
                        catch (...)
                        {
                            std::cout << "[FullProjectionDistance] Invalid epsilon '" << s << "', keeping "
                                      << opt.epsilon << ".\n";
                        }
                    };

                    std::string token;
                    while (iss >> token)
                    {
                        size_t eq = token.find('=');
                        if (eq != std::string::npos)
                        {
                            std::string key = token.substr(0, eq);
                            std::string val = token.substr(eq + 1);
                            if (key == "eps" || key == "epsilon" || key == "threshold")
                            {
                                parseEps(val);
                            }
                            else if (key == "same" || key == "same_anchor2")
                            {
                                opt.allowSameAnchorSecond = toBool(val);
                            }
                            else if (key == "norm" || key == "normalize")
                            {
                                opt.normalize = toBool(val);
                            }
                            else if (key == "reflect" || key == "reflection" || key == "mirror")
                            {
                                opt.allowReflection = toBool(val);
                            }
                            else if (key == "early" || key == "early_exit")
                            {
                                std::cout << "[FullProjectionDistance] '" << key
                                          << "' was removed: the outer matching is always computed.\n";
                            }
                            else
                            {
                                std::cout << "[FullProjectionDistance] Unknown key '" << key << "', ignored.\n";
                            }
                        }
                        else
                        {
                            parseEps(token);
                        }
                    }

                    molOps->printFullProjectionDistance(idx1, idx2, opt);
                }
                else
                {
                    auto tStart = std::chrono::high_resolution_clock::now();
                    float dist = molOps->computePairMetric(idx1, idx2, metric);
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
        "  Parameters for FullProjectionDistance (outer matching is always computed):\n"
        "    eps=<float>    : Strict < epsilon threshold (default: 0.01; also accepts bare float or epsilon=)\n"
        "                     Relative (dimensionless) when norm=1, absolute units when norm=0\n"
        "    norm=<0/1>     : 1 = normalized point-based representation NPR, divide by R^2 / R^3 (default: 1)\n"
        "    reflect=<0/1>  : 0 = chirality-sensitive, proper rotations only; 1 = also accept mirrored B (default: 0)\n"
        "    same=<0/1>     : 1 = allow a2==a1 and b2==b1; 0 = require distinct second anchor (default: 0)\n\n"
        "  Example:\n"
        "    pair_dist 4210 4214 aligner refine=4 symbol=1 exact=1\n"
        "    pair_dist 4210 4214 FullProjectionDistance eps=0.01 norm=1 reflect=0 same=0");
        ui.registerAlias("pairDist", "pair_dist");
        ui.registerAlias("dist_pair", "pair_dist");

        // dup_check
        ui.registerCommand(
            "dup_check", "<index|all> [epsilon|csv_file] [key=val ...]",
            "Checks for molecular duplicates against the DB.",
            [molOps](std::istringstream &iss)
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
                    auto matches = molOps->runBatchDuplicateCheck(csvPath, p, st);
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
                    auto matches = molOps->runDuplicateCheck(idx, csvPath, p, st);
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
            [molOps](std::istringstream &iss)
            {
                std::string kStr = "100%", metric = "srv", csvPath = "";
                if (iss >> kStr >> metric >> csvPath)
                {
                    molOps->searchKNNForAll(kStr, metric, csvPath);
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
            [molOps](std::istringstream &)
            {
                molOps->printDistanceDistribution();
            });
        ui.registerAlias("dist_stats", "dist_summary");
        ui.registerAlias("histogram", "dist_summary");

        // Register Command: view <index>
        ui.registerCommand(
            "view", "<index>",
            "Renders a specific loaded molecule as a colored point cloud in the Polyscope viewer.",
            [molOps](std::istringstream &iss)
            {
                size_t idx;
                if (iss >> idx)
                    molOps->visualizeIndex(idx);
                else
                    std::cout << "Usage: view <index>\n";
            });
        ui.registerAlias("visualize", "view");

        // Register Command: unview <index>
        ui.registerCommand(
            "unview", "<index>",
            "Removes a specific rendered molecule point cloud from the Polyscope viewer.",
            [molOps](std::istringstream &iss)
            {
                size_t idx;
                if (iss >> idx)
                    molOps->unviewIndex(idx);
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
                [molOps](size_t a, size_t b)
                { molOps->setMatchVisible(a, b, true); });
        pairCmd("hide_match", "Hides the matching lines.",
                [molOps](size_t a, size_t b)
                { molOps->setMatchVisible(a, b, false); });
        pairCmd("rm_match", "Deletes the stored matching for a pair from memory.",
                [molOps](size_t a, size_t b)
                { molOps->removeMatch(a, b); });
        pairCmd("apply_align", "Snaps cloud A onto cloud B using the stored rotation.",
                [molOps](size_t a, size_t b)
                { molOps->applyAlignment(a, b); });
        pairCmd("match_info", "Prints the bijection mapping (e.g. 0:1 1:3) and R/T matrices for a pair.",
                [molOps](size_t a, size_t b)
                { molOps->printMatchInfo(a, b); });

        ui.registerCommand("list_matches", "", "Lists all stored matchings.",
                           [](std::istringstream &)
                           { MatchingViz::list(); });
        ui.registerCommand("clear_matches", "", "Deletes all stored matchings.",
                           [](std::istringstream &)
                           { MatchingViz::clearAll(); });
        ui.registerCommand("show_matches", "", "Shows all stored matchings.",
                           [molOps](std::istringstream &)
                           { molOps->setAllMatchesVisible(true); });
        ui.registerCommand("hide_matches", "", "Hides all stored matchings.",
                           [molOps](std::istringstream &)
                           { molOps->setAllMatchesVisible(false); });

        // Cluster & Network Graph Commands
        ui.registerCommand(
            "explore_clusters", "",
            "Computes connected components from the last duplicate check and opens an interactive table window.",
            [molOps](std::istringstream &)
            { molOps->exploreClusters(); });
        ui.registerAlias("clusters", "explore_clusters");

        ui.registerCommand("show_graph", "", "Shows the duplicate network graph in Polyscope.",
                           [molOps](std::istringstream &)
                           {
                               molOps->setGraphVisible(true);
                               std::cout << "[Polyscope] Duplicate graph shown.\n";
                           });
        ui.registerCommand("hide_graph", "", "Hides the duplicate network graph in Polyscope.",
                           [molOps](std::istringstream &)
                           {
                               molOps->setGraphVisible(false);
                               std::cout << "[Polyscope] Duplicate graph hidden.\n";
                           });
        ui.registerCommand("clear_graph", "", "Removes the duplicate network graph and its labels.",
                           [molOps](std::istringstream &)
                           { molOps->clearGraph(); });
        ui.registerAlias("rm_graph", "clear_graph");

        // ComputeAtomicDescriptors
        ui.registerCommand(
            "ComputeAtomicDescriptors", "",
            "Computes centroid distance descriptors and sorted index arrays for all atomic clouds.",
            [atomOps](std::istringstream &)
            { atomOps->computeAtomicDescriptors(); });
        ui.registerAlias("compute_atoms", "ComputeAtomicDescriptors");

        // atomic_dist <index1> <index2> [metric: srv/aligner] [N=val] [key=val ...]
        ui.registerCommand(
            "atomic_dist", "<index1> <index2> [metric: srv/aligner] [N=val] [key=val ...]",
            "Computes Linf distance or rigid subset alignment between two atomic clouds.",
            [atomOps](std::istringstream &iss)
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
                        atomOps->computeAtomicAlignerDistance(idx1, idx2, N, opt);
                    }
                    else
                    {
                        auto tStart = std::chrono::high_resolution_clock::now();
                        float dist = atomOps->computeAtomicPairDistance(idx1, idx2, metric, N);
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
                           [](std::istringstream &iss)
                           {
                               size_t a, b;
                               if (iss >> a >> b)
                                   AtomicMatchingViz::applyAlignment(a, b);
                               else
                                   std::cout << "Usage: apply_atomic_align <idxA> <idxB>\n";
                           });

        ui.registerCommand("show_atomic_match", "<idxA> <idxB>", "Shows subset matching lines between two atomic clouds.",
                           [](std::istringstream &iss)
                           {
                               size_t a, b;
                               if (iss >> a >> b)
                                   AtomicMatchingViz::setVisible(a, b, true);
                               else
                                   std::cout << "Usage: show_atomic_match <idxA> <idxB>\n";
                           });
        ui.registerCommand("hide_atomic_match", "<idxA> <idxB>", "Hides subset matching lines.",
                           [](std::istringstream &iss)
                           {
                               size_t a, b;
                               if (iss >> a >> b)
                                   AtomicMatchingViz::setVisible(a, b, false);
                               else
                                   std::cout << "Usage: hide_atomic_match <idxA> <idxB>\n";
                           });

        ui.registerCommand(
            "atomic_express", "[epsilon] [N=val] [key=val ...] [verify] [trace] [quiet]",
            "Fast Stage-1-only SRV pair counter (optional CSV export + Stage-2 aligner). "
            "Prints per-label duplicate counts (C-O, C-N, ...). Supports batched pipelining, "
            "RAM tracking, and progress tracing.",
            [atomOps](std::istringstream &iss)
            {
                AtomicExpressParams p;
                bool verify = false;
                std::string token;

                auto toBool = [](const std::string &v)
                {
                    return v == "1" || v == "true" || v == "on" || v == "yes" ||
                           v == "TRUE" || v == "True" || v == "ON";
                };

                // convenience flags (no key= form)
                for (std::string tok; iss >> tok;)
                {
                    if (tok == "verify" || tok == "check" || tok == "oracle")
                    {
                        verify = true;
                        continue;
                    }
                    if (tok == "trace" || tok == "progress" || tok == "verbose")
                    {
                        p.traceProgress = true;
                        p.traceMemory = true;
                        p.traceVerbose = true;
                        continue;
                    }
                    if (tok == "trace-lite" || tok == "traceLite")
                    {
                        p.traceProgress = true;
                        continue;
                    }
                    if (tok == "quiet" || tok == "silent")
                    {
                        p.traceProgress = false;
                        p.traceMemory = false;
                        p.traceVerbose = false;
                        continue;
                    }
                    if (tok == "noviz" || tok == "no-viz" || tok == "nostore")
                    {
                        p.storeViz = false;
                        continue;
                    }

                    const size_t eq = tok.find('=');
                    if (eq != std::string::npos)
                    {
                        const std::string key = tok.substr(0, eq);
                        const std::string val = tok.substr(eq + 1);

                        // ---- Stage-1 core ----
                        if (key == "N" || key == "n")
                        {
                            try
                            {
                                p.N = std::stoul(val);
                            }
                            catch (...)
                            {
                            }
                        }
                        else if (key == "eps" || key == "epsilon")
                        {
                            try
                            {
                                p.epsilon = std::stof(val);
                            }
                            catch (...)
                            {
                            }
                        }
                        else if (key == "same")
                        {
                            p.considerSame = toBool(val);
                        }
                        else if (key == "srvf")
                        {
                            try
                            {
                                p.srvFactor = std::stof(val);
                            }
                            catch (...)
                            {
                            }
                        }
                        else if (key == "tol")
                        {
                            try
                            {
                                p.relTol = std::stof(val);
                            }
                            catch (...)
                            {
                            }
                        }
                        else if (key == "grid")
                        {
                            try
                            {
                                p.gridDim = std::stoi(val);
                            }
                            catch (...)
                            {
                            }
                        }
                        else if (key == "threads")
                        {
                            try
                            {
                                p.numThreads = std::stoi(val);
                            }
                            catch (...)
                            {
                            }
                        }
                        else if (key == "parallel")
                        {
                            p.parallel = toBool(val);
                        }
                        // ---- CSV export ----
                        else if (key == "csv" || key == "out" || key == "file")
                        {
                            p.csvPath = val;
                        }
                        else if (key == "names")
                        {
                            p.csvIncludeCloudNames = toBool(val);
                        }
                        else if (key == "counts")
                        {
                            p.csvIncludeAtomCounts = toBool(val);
                        }
                        // ---- Batching ----
                        else if (key == "batch" || key == "batchpairs" || key == "bp")
                        {
                            try
                            {
                                p.batchPairs = std::stoull(val);
                            }
                            catch (...)
                            {
                            }
                            if (p.batchPairs == 0)
                                p.autoTuneRam = false; // explicit 0 = disable
                        }
                        else if (key == "batchbytes" || key == "bb")
                        {
                            try
                            {
                                p.batchBytes = std::stoull(val);
                            }
                            catch (...)
                            {
                            }
                        }
                        else if (key == "auto" || key == "autotune" || key == "autoram")
                        {
                            p.autoTuneRam = toBool(val);
                        }
                        else if (key == "pipe" || key == "pipeline")
                        {
                            p.pipelineAlign = toBool(val);
                        }
                        else if (key == "topk" || key == "topkglobal")
                        {
                            p.topKGlobal = toBool(val);
                        }
                        // ---- Progress tracing ----
                        else if (key == "trace" || key == "progress")
                        {
                            p.traceProgress = toBool(val);
                        }
                        else if (key == "mem" || key == "rss")
                        {
                            p.traceMemory = toBool(val);
                        }
                        else if (key == "verbose" || key == "vtrace")
                        {
                            p.traceVerbose = toBool(val);
                        }
                        else if (key == "interval" || key == "dt")
                        {
                            try
                            {
                                p.traceIntervalSec = std::stod(val);
                            }
                            catch (...)
                            {
                            }
                        }
                        // ---- Stage-2 aligner ----
                        else if (key == "align" || key == "aligner" || key == "stage2")
                        {
                            p.runAligner = toBool(val);
                        }
                        else if (key == "optimal" || key == "opt" || key == "exact")
                        {
                            p.optimal = toBool(val);
                        }
                        else if (key == "alignf")
                        {
                            try
                            {
                                p.alignFactor = std::stof(val);
                            }
                            catch (...)
                            {
                            }
                        }
                        else if (key == "species")
                        {
                            p.requireSameNeighborSpecies = toBool(val);
                        }
                        else if (key == "refine")
                        {
                            try
                            {
                                p.refineIterations = std::stoi(val);
                            }
                            catch (...)
                            {
                            }
                        }
                        else if (key == "max")
                        {
                            try
                            {
                                p.maxMatches = std::stoul(val);
                            }
                            catch (...)
                            {
                            }
                        }
                        else if (key == "store" || key == "viz" || key == "storeviz")
                        {
                            p.storeViz = toBool(val);
                        }
                        else
                        {
                            std::cout << "[AtomicExpress] Unknown key '" << key << "', ignored.\n";
                        }
                    }
                    else if (tok.find(".csv") != std::string::npos)
                    {
                        p.csvPath = tok;
                    }
                    else
                    {
                        // bare number -> epsilon
                        try
                        {
                            p.epsilon = std::stof(tok);
                        }
                        catch (...)
                        {
                        }
                    }
                }

                // Guardrails
                if (p.maxMatches > 0 && !p.runAligner)
                {
                    std::cout << "[AtomicExpress] Note: 'max=' requires 'align=1'; ignoring maxMatches.\n";
                    p.maxMatches = 0;
                }
                if (!p.runAligner)
                    p.pipelineAlign = false; // no-op but makes banner honest
                if (p.batchPairs == 0 && p.batchBytes == 0 && !p.autoTuneRam)
                {
                    std::cout << "[AtomicExpress] Note: batching disabled — this materialises ALL "
                                 "Stage-1 survivors in RAM. Use batch=N or auto=1 to bound memory.\n";
                }

                atomOps->runAtomicExpressCount(p, verify);
            },
            "  Stage-1 (SRV) parameters:\n"
            "    epsilon         : Max SRV L_inf threshold in Angstroms (default: 0.10)\n"
            "    N=<int>         : Number of nearest non-center neighbor dims to compare (default: 4)\n"
            "    same=<0/1>      : 1 = compare all pairs; 0 = only different center types (default: 0)\n"
            "    srvf=<float>    : Stage-1 SRV radius multiplier factor (default: 1.0)\n"
            "    tol=<float>     : Floating-point noise slack tolerance (default: 1e-4)\n"
            "    grid=<int>      : Grid dimensionality cap, 1..4 (default: 4)\n"
            "    threads=<int>   : OpenMP thread override, 0 = max available (default: 0)\n"
            "    parallel=<0/1>  : Enable/disable multi-threaded execution (default: 1)\n\n"
            "  CSV export (optional, off unless csv= is given):\n"
            "    csv=<path>      : Write materialised pair list to this CSV (also accepts bare '*.csv')\n"
            "    names=<0/1>     : Include cloud names in CSV (default: 1)\n"
            "    counts=<0/1>    : Include atom counts in CSV (default: 1)\n\n"
            "  Batching (bounds peak RAM):\n"
            "    batch=<int>     : Manual pairs-per-batch (0 disables; default: auto)\n"
            "    batchbytes=<B>  : Manual bytes-per-batch (ignored if batch= is set)\n"
            "    auto=<0/1>      : Auto-size batches from free RAM (default: 1)\n"
            "    pipe=<0/1>      : Run Stage-2 aligner per-batch instead of deferring (default: 1)\n"
            "    topk=<0/1>      : 1 = global top-K at end (deterministic); 0 = per-batch (default: 1)\n\n"
            "  Progress tracing:\n"
            "    trace / trace-lite / quiet  : Convenience presets (no value needed)\n"
            "        trace       = progress + memory + verbose\n"
            "        trace-lite  = progress only\n"
            "        quiet       = all off (default)\n"
            "    trace=<0/1>     : Print phase markers, count/flush start/end (default: 0)\n"
            "    mem=<0/1>       : Append RSS to every trace line (default: 0)\n"
            "    verbose=<0/1>   : Per-table detail + every flush START line (default: 0)\n"
            "    interval=<s>    : Rate limit for %/ETA lines (default: 0.5 s)\n\n"
            "  Stage-2 aligner (optional, off unless align=1):\n"
            "    align=<0/1>     : Run bidirectional subset aligner on each Stage-1 survivor (default: 0)\n"
            "    optimal=<0/1>   : 1 = exact minimum bottleneck; 0 = fast early-exit (default: 0)\n"
            "    alignf=<float>  : Aligner bottleneck radius multiplier factor (default: 1.0)\n"
            "    species=<0/1>   : Require matched neighbor atoms to have identical element symbols (default: 0)\n"
            "    refine=<int>    : Kabsch refinement rounds (default: 0)\n"
            "    max=<int>       : Keep only top-K best aligned pairs, 0 = all (default: 0; requires align=1)\n\n"
            "  Viz store (default: on; only matters when align=1):\n"
            "    store=<0/1>     : 1 = push aligned pairs into AtomicMatchingViz (default: 1)\n"
            "                      0 = skip viz store (R.pairs still fully populated for CSV/report)\n"
            "    noviz           : Shorthand for store=0\n\n"
            "  Other:\n"
            "    verify          : Also run O(M^2) brute-force oracle and diff results (default: off)\n\n"
            "  Examples:\n"
            "    atomic_express\n"
            "    atomic_express 0.08 N=6 trace\n"
            "    atomic_express 0.10 same=1 grid=3 threads=16 trace mem\n"
            "    atomic_express 0.05 N=4 verify trace\n"
            "    atomic_express csv=pairs.csv trace mem\n"
            "    atomic_express csv=pairs.csv names=0 counts=0\n"
            "    atomic_express align=1 trace\n"
            "    atomic_express align=1 csv=verified.csv N=4 optimal=1 species=1 refine=4 max=200 trace\n"
            "    atomic_express align=1 alignf=2.0 optimal=0 threads=16 batch=1000000\n"
            "    atomic_express align=1 noviz csv=pairs.csv trace\n"
            "    atomic_express align=1 store=0 optimal=0 csv=pairs.csv\n"
            "    atomic_express auto=0 quiet   (legacy: materialise all, no progress)");
        ui.registerAlias("express", "atomic_express");
        ui.registerAlias("countPairs", "atomic_express");
        ui.registerAlias("atomicExpress", "atomic_express");
    }

} // namespace Commands