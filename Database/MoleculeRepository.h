#pragma once
#include "DatabaseConnection.h"
#include <vector>
#include <string>
#include <sstream>

class MoleculeRepository {
public:
    explicit MoleculeRepository(DatabaseConnection& dbConn) : m_db(dbConn) {}

    // Ensures the molecules table exists in MariaDB
    bool initializeSchema() {
        const char* query = 
            "CREATE TABLE IF NOT EXISTS molecules ("
            "  id INT AUTO_INCREMENT PRIMARY KEY,"
            "  filename VARCHAR(255) NOT NULL,"
            "  atom_count INT NOT NULL,"
            "  comment_line TEXT,"
            "  xyz_blob LONGBLOB NOT NULL,"
            "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ") ENGINE=InnoDB;";

        if (mysql_query(m_db.getHandle(), query)) {
            std::cerr << "[Repository] Failed to create table: " << mysql_error(m_db.getHandle()) << "\n";
            return false;
        }
        return true;
    }

    // Push a loaded local molecule into MariaDB
    bool saveMolecule(const std::string& filename, size_t atomCount, const std::string& comment, const std::string& rawXyzContent) {
        MYSQL* conn = m_db.getHandle();
        
        std::string query = "INSERT INTO molecules (filename, atom_count, comment_line, xyz_blob) VALUES (?, ?, ?, ?)";
        MYSQL_STMT* stmt = mysql_stmt_init(conn);
        if (!stmt) return false;

        if (mysql_stmt_prepare(stmt, query.c_str(), query.length())) {
            mysql_stmt_close(stmt);
            return false;
        }

        MYSQL_BIND bind[4];
        memset(bind, 0, sizeof(bind));

        // 1. filename
        bind[0].buffer_type = MYSQL_TYPE_STRING;
        bind[0].buffer = (void*)filename.c_str();
        bind[0].buffer_length = filename.length();

        // 2. atom_count
        bind[1].buffer_type = MYSQL_TYPE_LONG;
        bind[1].buffer = (void*)&atomCount;

        // 3. comment_line
        bind[2].buffer_type = MYSQL_TYPE_STRING;
        bind[2].buffer = (void*)comment.c_str();
        bind[2].buffer_length = comment.length();

        // 4. xyz_blob (Raw file text as blob)
        bind[3].buffer_type = MYSQL_TYPE_LONG_BLOB;
        bind[3].buffer = (void*)rawXyzContent.data();
        bind[3].buffer_length = rawXyzContent.length();

        if (mysql_stmt_bind_param(stmt, bind)) {
            mysql_stmt_close(stmt);
            return false;
        }

        if (mysql_stmt_execute(stmt)) {
            std::cerr << "[Repository] Insert failed: " << mysql_stmt_error(stmt) << "\n";
            mysql_stmt_close(stmt);
            return false;
        }

        mysql_stmt_close(stmt);
        return true;
    }

    // Pull all molecules from MariaDB blobs into your local memory store
    size_t loadAllIntoMemory(class MolecularDatasetStore& store) {
        MYSQL* conn = m_db.getHandle();
        if (mysql_query(conn, "SELECT filename, xyz_blob FROM molecules")) {
            std::cerr << "[Repository] Select failed: " << mysql_error(conn) << "\n";
            return 0;
        }

        MYSQL_RES* result = mysql_store_result(conn);
        if (!result) return 0;

        size_t loadedCount = 0;
        MYSQL_ROW row;
        unsigned long* lengths;

        while ((row = mysql_fetch_row(result))) {
            lengths = mysql_fetch_lengths(result);
            if (!row[1]) continue;

            std::string xyzData(row[1], lengths[1]);
            
            // Write to a temporary stream or parse directly into store
            // (You can add a method in MolecularDatasetStore to ingest raw string streams)
            // store.addStream(row[0], xyzData);
            loadedCount++;
        }

        mysql_free_result(result);
        std::cout << "[Repository] Pulled " << loadedCount << " molecules from MariaDB.\n";
        return loadedCount;
    }

private:
    DatabaseConnection& m_db;
};