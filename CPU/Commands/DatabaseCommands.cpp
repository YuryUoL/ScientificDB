#include "DatabaseCommands.h"
#include <iostream>

#if USE_MYSQL
namespace Commands {
void registerDatabaseCommands(CommandDispatcher &ui, MolecularDatasetStore &xyzStore, DatabaseConnection &dbConn) {
    // db_connect
    ui.registerCommand(
        "db_connect", "<host> <user> <pass> <db>",
        "Connects the client to a remote or local MariaDB server.",
        [&dbConn](std::istringstream &iss) {
            std::string host, user, pass, db;
            if (iss >> host >> user >> pass >> db) {
                if (dbConn.connect(host, user, pass, db)) {
                    MoleculeRepository repo(dbConn);
                    repo.initializeSchema();
                    std::cout << "[Database] Connected and schema verified.\n";
                }
            } else {
                std::cout << "Usage: db_connect <host> <user> <pass> <db>\n";
            }
        });

    // db_disconnect
    ui.registerCommand(
        "db_disconnect", "",
        "Disconnects the client from the active MariaDB server.",
        [&dbConn](std::istringstream &) {
            if (!dbConn.isConnected()) {
                std::cout << "[Database] Not currently connected to any server.\n";
                return;
            }
            dbConn.disconnect();
            std::cout << "[Database] Disconnected.\n";
        });
    ui.registerAlias("db_quit", "db_disconnect");

    // db_push
    ui.registerCommand(
        "db_push", "[include_descriptors: 0/1]",
        "Uploads all currently loaded local RAM molecules into MariaDB blobs.",
        [&dbConn, &xyzStore](std::istringstream &iss) {
            if (!dbConn.isConnected()) {
                std::cout << "[Database] Error: Not connected. Run 'db_connect' first.\n";
                return;
            }
            int includeDesc = 1;
            iss >> includeDesc;

            std::cout << "[Database] Pushing local dataset to server...\n";
            MoleculeRepository repo(dbConn);
            if (!repo.beginTransaction()) {
                std::cerr << "[Database] Failed to start transaction.\n";
                return;
            }

            size_t successCount = 0;
            size_t total = xyzStore.size();
            for (size_t i = 0; i < total; ++i) {
                std::string filename = xyzStore.getFilename(i);
                std::string rawXyz = xyzStore.getRawContent(i);
                std::string comment = xyzStore.getComment(i);
                size_t atomCount = xyzStore.getAtomCount(i);

                std::vector<float> sdv, srv, pdd;
                if (includeDesc) {
                    sdv = xyzStore.getSDV(i);
                    srv = xyzStore.getSRV(i);
                    pdd = xyzStore.getPDD(i);
                }

                if (repo.saveMolecule(filename, atomCount, comment, rawXyz, sdv, srv, pdd)) {
                    successCount++;
                }
            }

            if (successCount == total) {
                repo.commit();
                std::cout << "[Database] Successfully pushed and committed " << successCount << " / " << total << " molecules.\n";
            } else {
                repo.rollback();
                std::cout << "[Database] Error during push. Transaction rolled back.\n";
            }
        });

    // db_pull
    ui.registerCommand(
        "db_pull", "[include_descriptors: 0/1]",
        "Downloads molecule blobs from MariaDB into local RAM store.",
        [&dbConn, &xyzStore](std::istringstream &iss) {
            if (!dbConn.isConnected()) {
                std::cout << "[Database] Error: Not connected.\n";
                return;
            }
            int includeDesc = 1;
            iss >> includeDesc;
            MoleculeRepository repo(dbConn);
            size_t pulled = repo.loadAllIntoMemory(xyzStore, (includeDesc != 0));
            std::cout << "[Database] Pulled " << pulled << " molecules into local RAM.\n";
        });

    // db_clear
    ui.registerCommand(
        "db_clear", "",
        "Truncates and clears all molecules stored in MariaDB.",
        [&dbConn](std::istringstream &) {
            if (!dbConn.isConnected()) {
                std::cout << "[Database] Error: Not connected.\n";
                return;
            }
            MoleculeRepository repo(dbConn);
            if (repo.clearDatabase()) {
                std::cout << "[Database] Successfully cleared MariaDB database.\n";
            }
        });
}
} // namespace Commands
#endif