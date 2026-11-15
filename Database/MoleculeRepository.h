#pragma once
#include "DatabaseConnection.h"
#include <vector>
#include <string>
#include <sstream>
#include <cstring>

class MoleculeRepository {
public:
    explicit MoleculeRepository(DatabaseConnection& dbConn) : m_db(dbConn) {}

    bool initializeSchema() {
        const char* query = 
            "CREATE TABLE IF NOT EXISTS molecules ("
            "  id INT AUTO_INCREMENT PRIMARY KEY,"
            "  filename VARCHAR(255) NOT NULL,"
            "  atom_count INT NOT NULL,"
            "  comment_line TEXT,"
            "  xyz_blob LONGBLOB NOT NULL,"
            "  sdv_blob LONGBLOB DEFAULT NULL,"
            "  srv_blob LONGBLOB DEFAULT NULL,"
            "  pdd_blob LONGBLOB DEFAULT NULL,"
            "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ") ENGINE=InnoDB;";

        if (mysql_query(m_db.getHandle(), query)) {
            std::cerr << "[Repository] Failed to create table: " << mysql_error(m_db.getHandle()) << "\n";
            return false;
        }
        return true;
    }
    // Clears all stored molecules from the database instantly
    bool clearDatabase() {
        MYSQL* conn = m_db.getHandle();
        const char* query = "TRUNCATE TABLE molecules;";
        
        if (mysql_query(conn, query)) {
            std::cerr << "[Repository] Failed to clear database table: " << mysql_error(conn) << "\n";
            return false;
        }
        return true;
    }

    bool saveMolecule(const std::string& filename, size_t atomCount, const std::string& comment, 
                      const std::string& rawXyzContent,
                      const std::vector<float>& sdv = {},
                      const std::vector<float>& srv = {},
                      const std::vector<float>& pdd = {}) 
    {
        MYSQL* conn = m_db.getHandle();
        
        std::string query = "INSERT INTO molecules (filename, atom_count, comment_line, xyz_blob, sdv_blob, srv_blob, pdd_blob) VALUES (?, ?, ?, ?, ?, ?, ?)";
        MYSQL_STMT* stmt = mysql_stmt_init(conn);
        if (!stmt) return false;

        if (mysql_stmt_prepare(stmt, query.c_str(), query.length())) {
            std::cerr << "[Repository] Prepare failed: " << mysql_stmt_error(stmt) << "\n";
            mysql_stmt_close(stmt);
            return false;
        }

        MYSQL_BIND bind[7];
        memset(bind, 0, sizeof(bind));

        static my_bool sdv_is_null = 0;
        static my_bool srv_is_null = 0;
        static my_bool pdd_is_null = 0;
        static my_bool comment_is_null = 0;

        bind[0].buffer_type = MYSQL_TYPE_STRING;
        bind[0].buffer = (void*)filename.c_str();
        bind[0].buffer_length = filename.length();

        int32_t atomCount32 = static_cast<int32_t>(atomCount);
        bind[1].buffer_type = MYSQL_TYPE_LONG;
        bind[1].buffer = (void*)&atomCount32;

        comment_is_null = comment.empty() ? 1 : 0;
        bind[2].buffer_type = MYSQL_TYPE_STRING;
        bind[2].buffer = (void*)comment.c_str();
        bind[2].buffer_length = comment.length();
        bind[2].is_null = &comment_is_null;

        bind[3].buffer_type = MYSQL_TYPE_LONG_BLOB;
        bind[3].buffer = (void*)rawXyzContent.data();
        bind[3].buffer_length = rawXyzContent.length();

        sdv_is_null = sdv.empty() ? 1 : 0;
        bind[4].buffer_type = MYSQL_TYPE_LONG_BLOB;
        bind[4].buffer = sdv.empty() ? nullptr : (void*)sdv.data();
        bind[4].buffer_length = sdv.empty() ? 0 : sdv.size() * sizeof(float);
        bind[4].is_null = &sdv_is_null;

        srv_is_null = srv.empty() ? 1 : 0;
        bind[5].buffer_type = MYSQL_TYPE_LONG_BLOB;
        bind[5].buffer = srv.empty() ? nullptr : (void*)srv.data();
        bind[5].buffer_length = srv.empty() ? 0 : srv.size() * sizeof(float);
        bind[5].is_null = &srv_is_null;

        pdd_is_null = pdd.empty() ? 1 : 0;
        bind[6].buffer_type = MYSQL_TYPE_LONG_BLOB;
        bind[6].buffer = pdd.empty() ? nullptr : (void*)pdd.data();
        bind[6].buffer_length = pdd.empty() ? 0 : pdd.size() * sizeof(float);
        bind[6].is_null = &pdd_is_null;

        if (mysql_stmt_bind_param(stmt, bind)) {
            std::cerr << "[Repository] Bind failed: " << mysql_stmt_error(stmt) << "\n";
            mysql_stmt_close(stmt);
            return false;
        }

        if (mysql_stmt_execute(stmt)) {
            std::cerr << "[Repository] Insert failed for file '" << filename 
                      << "': " << mysql_stmt_error(stmt) << " (Errno: " << mysql_stmt_errno(stmt) << ")\n";
            mysql_stmt_close(stmt);
            return false;
        }

        mysql_stmt_close(stmt);
        return true;
    }

    size_t loadAllIntoMemory(class MolecularDatasetStore& store, bool includeDescriptors = true) {
        MYSQL* conn = m_db.getHandle();
        
        std::string query = "SELECT filename, xyz_blob FROM molecules";

        if (mysql_query(conn, query.c_str())) {
            std::cerr << "[Repository] Select failed: " << mysql_error(conn) << "\n";
            return 0;
        }

        MYSQL_RES* result = mysql_store_result(conn);
        if (!result) return 0;

        size_t loadedCount = 0;
        MYSQL_ROW row;
        unsigned long* lengths;

        store.clearAll();

        while ((row = mysql_fetch_row(result))) {
            lengths = mysql_fetch_lengths(result);
            if (!row[1]) continue;

            std::string filename(row[0]);
            std::string xyzData(row[1], lengths[1]);
            
            if (store.addStringContent(filename, xyzData)) {
                loadedCount++;
            }
        }

        mysql_free_result(result);
        std::cout << "[Repository] Pulled " << loadedCount << " molecules from MariaDB into local RAM.\n";
        
        if (includeDescriptors && loadedCount > 0) {
            std::cout << "[XyzStore] Automatically re-computing descriptors for pulled dataset...\n";
            store.computeDescriptors();
        }

        return loadedCount;
    }

    bool beginTransaction() {
        return mysql_query(m_db.getHandle(), "START TRANSACTION") == 0;
    }

    bool commit() {
        return mysql_query(m_db.getHandle(), "COMMIT") == 0;
    }

    bool rollback() {
        return mysql_query(m_db.getHandle(), "ROLLBACK") == 0;
    }

private:
    DatabaseConnection& m_db;
};