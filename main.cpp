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

// Thread-safe command queue & synchronization primitives
static std::mutex g_cmdMutex;
static std::condition_variable g_cmdCV;
static std::queue<std::string> g_cmdQueue;
static std::atomic<bool> g_running{true};
static bool g_readyForInput{true};

/**
 * @brief Asynchronous CLI reader thread maintaining terminal line-editing & command history via GNU Readline.
 */
void inputThreadWorker() {
    char *rawInput = nullptr;
    while (g_running) {
        {
            std::unique_lock<std::mutex> lock(g_cmdMutex);
            g_cmdCV.wait(lock, [] { return g_readyForInput || !g_running; });
            if (!g_running) break;
        }

        rawInput = readline("> ");
        if (!rawInput) { // Handles Ctrl+D (EOF)
            g_running = false;
            break;
        }

        std::string line(rawInput);
        if (!line.empty()) {
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
        if (cmd == "quit" || cmd == "exit") {
            g_running = false;
            break;
        }
    }
    g_running = false;
}

int main(int argc, char* argv[]) {
    std::cout << "Initializing Polyscope GUI...\n";
    PolyscopeBridge::init();

    MolecularDatasetStore xyzStore;
    CommandDispatcher ui;

    // Register Command: loadXyz <file_or_folder>
    ui.registerCommand(
        "loadXyz", "<path_to_file_or_folder>",
        "Loads single .xyz file or a whole folder of .xyz files into host RAM.",
        [&](std::istringstream &iss) {
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

    // Register Command: ShowAll [limit]
    ui.registerCommand(
        "ShowAll", "[limit]",
        "Lists all loaded .xyz molecules with general atom count info.",
        [&](std::istringstream &iss) {
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
        [&](std::istringstream &iss) {
            size_t idx;
            if (iss >> idx) {
                xyzStore.inspect(idx);
            } else {
                std::cout << "Usage: check <index>\n";
            }
        });
    ui.registerAlias("inspectXyz", "check");

    // Register Command: removeXyz <index>
    ui.registerCommand(
        "removeXyz", "<index>",
        "Removes a molecule by its list index.",
        [&](std::istringstream &iss) {
            size_t idx;
            if (iss >> idx) {
                xyzStore.removeByIndex(idx);
            } else {
                std::cout << "Usage: removeXyz <index>\n";
            }
        });
    ui.registerAlias("rmXyz", "removeXyz");

    // Register Command: clearXyz
    ui.registerCommand(
        "clearXyz", "",
        "Clears all loaded molecular datasets.",
        [&](std::istringstream &) {
            xyzStore.clearAll();
        });

    // Register Command: view <index>
    ui.registerCommand(
        "view", "<index>",
        "Renders a specific loaded molecule as a colored point cloud in the Polyscope viewer.",
        [&](std::istringstream &iss) {
            size_t idx;
            if (iss >> idx) {
                xyzStore.visualizeIndex(idx);
            } else {
                std::cout << "Usage: view <index>\n";
            }
        });
    ui.registerAlias("visualize", "view");

    // Register Command: gizmo on/off
    ui.registerCommand(
        "gizmo", "<on/off>",
        "Enables or disables interactive transform gizmos for the point clouds.",
        [&](std::istringstream &iss) {
            std::string state;
            iss >> state;
            if (state == "on" || state == "1") {
                PolyscopeBridge::setGizmoEnabled(true);
                std::cout << "[Polyscope] Interactive gizmos enabled.\n";
            } else if (state == "off" || state == "0") {
                PolyscopeBridge::setGizmoEnabled(false);
                std::cout << "[Polyscope] Interactive gizmos disabled.\n";
            } else {
                std::cout << "Usage: gizmo on  OR  gizmo off\n";
            }
        });

    // Register Command: help
    ui.registerCommand("help", "", "Displays this command usage reference.",
                       [&](std::istringstream &) {
                           ui.printHelp();
                       });

    // Register Command: ComputeDescriptors
    ui.registerCommand(
        "ComputeDescriptors", "",
        "Computes SRD and SRV descriptors in parallel for all loaded molecules using OpenMP.",
        [&](std::istringstream &) {
            xyzStore.computeDescriptors();
        });
    ui.registerAlias("compute", "ComputeDescriptors");

    // Register Command: FlushDescriptors
    ui.registerCommand(
        "FlushDescriptors", "",
        "Deletes and flushes all computed descriptors and FAISS indices.",
        [&](std::istringstream &) {
            xyzStore.flushDescriptors();
        });
    ui.registerAlias("flush", "FlushDescriptors");

    // Register Command: unview <index>
    ui.registerCommand(
        "unview", "<index>",
        "Removes a specific rendered molecule point cloud from the Polyscope viewer.",
        [&](std::istringstream &iss) {
            size_t idx;
            if (iss >> idx) {
                xyzStore.unviewIndex(idx);
            } else {
                std::cout << "Usage: unview <index>\n";
            }
        });
    ui.registerAlias("hide", "unview");
    ui.registerAlias("unvisualize", "unview");

    // Register Command: quit / exit
    ui.registerCommand("quit", "", "Exits the application.",
                       [&](std::istringstream &) {
                           g_running = false;
                       });
    ui.registerAlias("exit", "quit");

    if (argc >= 2) {
        std::string initialPath = argv[1];
        namespace fs = std::filesystem;
        if (fs::is_directory(initialPath)) {
            xyzStore.addFolder(initialPath);
        } else if (fs::is_regular_file(initialPath)) {
            xyzStore.addFile(initialPath);
        }
    }

    std::cout << "\nInteractive Molecular Store CLI Active.\n";
    std::cout << "Type 'help' for commands or 'ShowAll' to view loaded entries.\n\n";

    // Launch non-blocking CLI input thread
    std::thread inputThread(inputThreadWorker);

    // Main non-blocking render & event loop
    while (g_running) {
        std::string line;
        bool haveLine = false;

        {
            std::lock_guard<std::mutex> lock(g_cmdMutex);
            if (!g_cmdQueue.empty()) {
                line = g_cmdQueue.front();
                g_cmdQueue.pop();
                haveLine = true;
            }
        }

        if (haveLine) {
            ui.execute(line);

            {
                std::lock_guard<std::mutex> lock(g_cmdMutex);
                g_readyForInput = true;
            }
            g_cmdCV.notify_one();
        }

        // Keep Polyscope responsive continuously
        PolyscopeBridge::drawFrame();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::cout << "Cleaning up and exiting...\n";
    if (inputThread.joinable()) {
        inputThread.detach();
    }

    return 0;
}