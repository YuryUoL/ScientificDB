#pragma once
#include "CPU/CommandDispatcher.h"
#include "CPU/MolecularDatasetStore.h"

#if USE_MYSQL
#include "Database/DatabaseConnection.h"
#include "Database/MoleculeRepository.h"
#endif

namespace Commands {
#if USE_MYSQL
    void registerDatabaseCommands(CommandDispatcher &ui, MolecularDatasetStore &xyzStore, DatabaseConnection &dbConn);
#else
    inline void registerDatabaseCommands(CommandDispatcher &, MolecularDatasetStore &, void *) {}
#endif
}