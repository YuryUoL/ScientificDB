#pragma once
#include "CPU/CommandDispatcher.h"
#include "CPU/MolecularDatasetStore.h"

namespace Commands {
    void registerDatasetCommands(CommandDispatcher &ui, MolecularDatasetStore &xyzStore);
}