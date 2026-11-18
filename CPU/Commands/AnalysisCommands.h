#pragma once
#include "CPU/CommandDispatcher.h"
#include "CPU/MolecularDatasetStore.h"

namespace Commands {
    void registerAnalysisCommands(CommandDispatcher &ui, MolecularDatasetStore &xyzStore);
}