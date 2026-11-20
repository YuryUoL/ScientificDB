#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <filesystem>

// GNU Readline headers
#include <readline/readline.h>
#include <readline/history.h>

#include "CPU/CommandDispatcher.h"
#include "CPU/MolecularDatasetStore.h"
#include "CPU/MolecularDatasetStore.h"
#include "CPU/MolecularOperations.h"

#include "CPU/Visualization/AtomicMatchingViz.h"
#include "CPU/Visualization/PolyscopeBridge.h"
#include "CPU/Visualization/Matchingvisualizer.h"
#include "CPU/Visualization/DatasetBrowserViz.h"
#include "CPU/Visualization/AtomicBrowserViz.h"

// Modular Command Groups
#include "CPU/Commands/DatasetCommands.h"
#include "CPU/Commands/AnalysisCommands.h"
#include "CPU/Commands/DatabaseCommands.h"

#if USE_MYSQL
#include "Database/DatabaseConnection.h"
static DatabaseConnection g_dbConn;
#endif

// Thread-safe command queue & synchronization primitives
static std::mutex g_cmdMutex;
static std::condition_variable g_cmdCV;
static std::queue<std::string> g_cmdQueue;
static std::atomic<bool> g_running{true};
static bool g_readyForInput{true};

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
        {
            g_running = false;
            break;
        }

        std::string line(rawInput);
        if (!line.empty())
        {
            add_history(rawInput);
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

    DatasetBrowserViz::s_windowVisible = false;

    PolyscopeBridge::s_extraUserCallback = [&]()
    {
        ImGui::Checkbox("Dataset Browser", &DatasetBrowserViz::s_windowVisible);
        ImGui::SameLine();
        ImGui::Checkbox("Atomic Browser", &AtomicBrowserViz::s_windowVisible);

        // Only allocate if the cluster graph window is actually open
        static std::vector<std::string> emptyFilenames;
        static std::vector<size_t> emptyAtomCounts;
        MolecularGraphViz::drawImGuiWindow(emptyFilenames, emptyAtomCounts);

        if (DatasetBrowserViz::s_windowVisible)
        {
            DatasetBrowserViz::drawWindow(xyzStore);
        }

        if (AtomicBrowserViz::s_windowVisible)
        {
            AtomicBrowserViz::drawWindow(xyzStore);
        }
    };

    MatchingViz::install();
    AtomicMatchingViz::install(); // <--- Add this line

    // Register Modular Commands
    Commands::registerDatasetCommands(ui, xyzStore);
    Commands::registerAnalysisCommands(ui, xyzStore);

#if USE_MYSQL
    Commands::registerDatabaseCommands(ui, xyzStore, g_dbConn);
#endif

    // Basic GUI toggle & help/quit commands
    ui.registerCommand("browser", "<on/off>", "Shows or hides the dataset browser window.", [&](std::istringstream &iss)
                       {
        std::string state;
        if (iss >> state) {
            DatasetBrowserViz::s_windowVisible = (state == "on" || state == "1");
        } else {
            std::cout << "Dataset Browser is currently: " << (DatasetBrowserViz::s_windowVisible ? "ON" : "OFF") << "\n";
        } });

    ui.registerCommand("help", "[command_name]", "Displays general help or detailed options for a specific command.", [&](std::istringstream &iss)
                       {
        std::string cmdName;
        if (iss >> cmdName) {
            ui.printHelp(cmdName);
        } else {
            ui.printHelp();
        } });
    ui.registerCommand("quit", "", "Exits the application.", [&](std::istringstream &)
                       { g_running = false; });
    ui.registerAlias("exit", "quit");

  if (argc >= 2)
    {
        std::string initialPath = argv[1];
        bool autoCompute = (argc >= 3 && (std::string(argv[2]) == "--compute" || std::string(argv[2]) == "-c"));
        namespace fs = std::filesystem;
        if (fs::is_directory(initialPath))
        {
            xyzStore.addFolder(initialPath);
        }
        else if (fs::is_regular_file(initialPath))
        {
            xyzStore.addFile(initialPath);
        }
        if (autoCompute)
        {
            MolecularOperations molOps(xyzStore);
            molOps.computeDescriptors();
        }
    }

    std::cout << "\nInteractive Molecular Store CLI Active.\nType 'help' for commands.\n\n";

    std::thread inputThread(inputThreadWorker);

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

    if (inputThread.joinable())
    {
        inputThread.detach();
    }
    return 0;
}